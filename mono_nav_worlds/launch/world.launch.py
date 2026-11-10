"""Start Gazebo Fortress with one of the benchmark worlds and spawn mono_bot.

usage:
  ros2 launch ./launch/world.launch.py world:=test_01
  ros2 launch ./launch/world.launch.py world:=small_house x:=0 y:=0
  ros2 launch ./launch/world.launch.py world:=train_02 urdf:=/path/to/mono_bot.urdf

world: a name from worlds/ (without .sdf) or a full path to an .sdf file.
The default start pose (0.7, 0, 0) matches the generated corridor worlds.
"""
import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URDF = os.path.expanduser('~/depth_safe_robot/mono_nav_worlds/mono_bot.urdf')


def setup(context):
    world = LaunchConfiguration('world').perform(context)
    if not world.endswith('.sdf'):
        world = os.path.join(ROOT, 'worlds', world + '.sdf')
    urdf = LaunchConfiguration('urdf').perform(context)
    with open(urdf) as f:
        robot_desc = f.read()

    model_paths = [os.path.join(ROOT, 'models'),
                   os.path.join(ROOT, 'third_party', 'aws-robomaker-small-house-world', 'models')]
    old = os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    res_path = ':'.join(model_paths + ([old] if old else []))

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r {world}'}.items())

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'robot_description': robot_desc, 'use_sim_time': True}])

    spawn = Node(package='ros_gz_sim', executable='create', arguments=[
        '-topic', 'robot_description', '-name', 'mono_bot',
        '-x', LaunchConfiguration('x').perform(context),
        '-y', LaunchConfiguration('y').perform(context),
        '-z', '0.06',
        '-Y', LaunchConfiguration('yaw').perform(context)])

    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', arguments=[
        '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
        '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
        '/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
        '/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model',
        '/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
        '/camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
        '/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
    ], parameters=[{'use_sim_time': True}])

    return [SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', res_path),
            gazebo, rsp, spawn, bridge]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='test_clean'),
        DeclareLaunchArgument('urdf', default_value=DEFAULT_URDF),
        DeclareLaunchArgument('x', default_value='0.7'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        OpaqueFunction(function=setup),
    ])
