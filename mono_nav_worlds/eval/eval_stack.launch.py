"""Whole stack for one world + one method in a single launch (same as your 7 terminals).

  ros2 launch ./eval_stack.launch.py world:=test_02 method:=gt
  ros2 launch ./eval_stack.launch.py world:=test_02 method:=mono rviz:=true
  ros2 launch ./eval_stack.launch.py world:=train_01 method:=gt collect:=true

method:=gt    point cloud from /camera/depth_image (or /gt/depth with gt_sync:=true) -> /points
method:=mono  point cloud from /mono/depth -> /mono/points
method:=mono_margin | mono_mu | mono_safe   same, but /mono/depth comes from mono_depth_safe
              (uncertainty head; head_ckpt:=runs/head_v1/head_best.pt)
method:=mono_safe_k1.0 / mono_margin_c0.2   sweep variants with an explicit k or c
mono_depth runs in BOTH methods so GPU load is identical.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            OpaqueFunction, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = {
    'world': ('test_02', 'world name (worlds/<world>.sdf)'),
    'method': ('gt', 'gt | mono | mono_margin | mono_mu | mono_safe'),
    'head_ckpt': (os.path.join(HERE, 'runs', 'head_v1', 'head_best.pt'), 'uncertainty head checkpoint'),
    'debug': ('false', 'mono_depth_safe also publishes /mono/sigma etc.'),
    'floor_snap': ('true', 'snap floor pixels to the floor plane (all mono methods)'),
    'floor_z': ('-0.05', 'height of the real floor in the odom frame (odom z=0 is base_link level)'),
    'robot': ('mono_bot', 'robot model name in Gazebo'),
    'worlds_dir': ('~/depth_safe_robot/mono_nav_worlds', 'folder with launch/ and worlds/'),
    'nav2_params': ('', 'default: <worlds_dir>/ground_depth_nav2.yaml or mono_depth_nav2.yaml'),
    'da_repo': (os.environ.get('DA', ''), 'Depth-Anything-V2 repo (default $DA)'),
    'checkpoint': ('', 'default: <da_repo>/checkpoints/depth_anything_v2_metric_hypersim_vits.pth'),
    'input_size': ('518', 'Depth Anything input size (364 is ~2x faster)'),
    'gt_sync': ('false', 'GT cloud from /gt/depth (same frames/latency as mono)'),
    'compare_depth': ('false', 'also run compare_depth'),
    'rviz': ('false', 'start RViz'),
    'collect': ('false', 'run collect_data.py'),
    'dataset': (os.path.join(HERE, 'dataset'), 'collect_data output folder'),
    'nav_delay': ('15.0', 's to wait before starting Nav2'),
}


def setup(context):
    g = {k: LaunchConfiguration(k).perform(context) for k in ARGS}
    on = lambda k: g[k].lower() in ('1', 'true', 'yes')
    wd = os.path.expanduser(g['worlds_dir'])
    method = g['method']
    da = os.path.expanduser(g['da_repo'])
    ckpt = g['checkpoint'] or os.path.join(da, 'checkpoints', 'depth_anything_v2_metric_hypersim_vits.pth')
    nav_params = g['nav2_params'] or os.path.join(
        wd, 'ground_depth_nav2.yaml' if method == 'gt' else 'mono_depth_nav2.yaml')
    sim = {'use_sim_time': True}

    acts = [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(wd, 'launch', 'world.launch.py')),
        launch_arguments={'world': g['world']}.items())]

    # bumper: Gazebo publishes contacts on its default topic, bridge it to /bumper
    gz_contact = (f"/world/{g['world']}/model/{g['robot']}/link/base_link/sensor/bumper/contact")
    acts.append(Node(package='ros_gz_bridge', executable='parameter_bridge', name='bumper_bridge',
                     arguments=[f'{gz_contact}@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts'],
                     remappings=[(gz_contact, '/bumper')], parameters=[sim]))

    head_modes = {'mono': 'raw', 'mono_margin': 'margin', 'mono_mu': 'mu', 'mono_safe': 'safe'}
    # sweep variants: mono_safe_k2.0 -> safe with k=2.0 ; mono_margin_c0.3 -> margin with c=0.3
    k_over, c_over, base = -1.0, -1.0, method
    if method.startswith('mono_safe_k'):
        base, k_over = 'mono_safe', float(method[len('mono_safe_k'):])
    elif method.startswith('mono_margin_c'):
        base, c_over = 'mono_margin', float(method[len('mono_margin_c'):])
    if base in head_modes:
        # all mono methods use the same node -> identical preprocessing and floor snapping
        if method != 'mono' and not os.path.exists(os.path.expanduser(g['head_ckpt'])):
            raise RuntimeError(f'method {method} needs head_ckpt:=.../head_best.pt')
        acts.append(Node(package='mono_nav', executable='mono_depth_safe', output='screen',
                         parameters=[{'repo_path': da, 'checkpoint': ckpt,
                                      'input_size': int(g['input_size']),
                                      'head_ckpt': os.path.expanduser(g['head_ckpt']),
                                      'mode': head_modes[base],
                                      'k': k_over, 'margin_c': c_over,
                                      'floor_snap': on('floor_snap'),
                                      'floor_z': float(g['floor_z']),
                                      'publish_debug': on('debug')}]))
    else:     # gt: plain DA2 runs too, so the GPU load is the same
        acts.append(Node(package='mono_nav', executable='mono_depth', output='screen',
                         parameters=[{'repo_path': da, 'checkpoint': ckpt,
                                      'input_size': int(g['input_size'])}]))
    if on('compare_depth'):
        acts.append(Node(package='mono_nav', executable='compare_depth'))

    if method == 'gt':
        depth, info, points, cloud = '/camera/depth_image', '/camera/camera_info', '/points', 'gt_cloud'
        if on('gt_sync'):
            acts.append(Node(package='mono_nav', executable='gt_sync', parameters=[sim]))
            depth = '/gt/depth'
    else:
        # BUG FIX: mono depth arrives ~1 s after its camera frame. depth_image_proc pairs depth
        # and camera_info by EXACT stamp with a short queue, so /camera/camera_info was usually
        # gone -> no point cloud -> robot drove blind in batch runs (RViz slowed the sim enough
        # to hide it in manual runs). mono_depth_safe republishes the matching camera_info.
        depth, info, points, cloud = '/mono/depth', '/mono/camera_info', '/mono/points', 'mono_cloud'
    acts.append(Node(package='depth_image_proc', executable='point_cloud_xyz_node', name=cloud,
                     remappings=[('image_rect', depth), ('camera_info', info),
                                 ('points', points)],
                     parameters=[sim, {'queue_size': 30}]))
    acts.append(Node(package='mono_nav', executable='odom_to_tf', parameters=[sim]))

    if on('rviz'):
        acts.append(Node(package='rviz2', executable='rviz2', parameters=[sim], arguments=[
            '-d', os.path.join(get_package_share_directory('nav2_bringup'), 'rviz',
                               'nav2_default_view.rviz')]))

    acts.append(TimerAction(period=float(g['nav_delay']), actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')),
        launch_arguments={'params_file': nav_params, 'use_sim_time': 'True'}.items())]))

    if on('collect'):
        acts.append(ExecuteProcess(cmd=['python3', os.path.join(HERE, 'collect_data.py'),
                                        '--out', os.path.expanduser(g['dataset']),
                                        '--method', method], output='screen'))
    return acts


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v, description=d) for k, (v, d) in ARGS.items()]
        + [OpaqueFunction(function=setup)])
