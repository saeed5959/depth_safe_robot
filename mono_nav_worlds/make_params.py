#!/usr/bin/env python3
"""Build a mapless, camera-only Nav2 config for mono_bot (ROS 2 Humble).

Starts from the official Humble nav2_params.yaml and changes only what matters:
  - local + global costmaps use a VoxelLayer fed by ONE point-cloud topic
  - everything runs in the odom frame (no map, no AMCL)
  - softer inflation, bigger local costmap (camera sees ~4 m)
  - more patient progress checker, 10 Hz controller
  - a custom behavior tree with short recoveries (camera_bt.xml, same folder)

usage:
  python3 make_params.py                         # ground-truth depth  (/points)
  python3 make_params.py --cloud /mono/points    # Depth Anything raw
  python3 make_params.py --cloud /safe/points    # your method (mu - 2 sigma)
  python3 make_params.py --cloud /mono/points --out mono_params.yaml
"""
import argparse
import copy
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = '/opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml'

ap = argparse.ArgumentParser()
ap.add_argument('--cloud', default='/points', help='PointCloud2 topic for the costmaps')
ap.add_argument('--out', default=os.path.join(HERE, 'my_nav2_params.yaml'))
ap.add_argument('--robot-radius', type=float, default=0.25)
a = ap.parse_args()

p = yaml.safe_load(open(SRC))

# ------------------------------------------------------------------ costmap layers
voxel = {
    'plugin': 'nav2_costmap_2d::VoxelLayer',
    'enabled': True,
    'publish_voxel_map': True,
    'footprint_clearing_enabled': True,
    'origin_z': 0.0, 'z_resolution': 0.05, 'z_voxels': 16,
    'max_obstacle_height': 0.8,
    'mark_threshold': 0,
    'unknown_threshold': 15,
    'combination_method': 1,
    'observation_sources': 'depth_cloud',
    'depth_cloud': {
        'topic': a.cloud,                 # <- the only line that differs between methods
        'data_type': 'PointCloud2',
        'marking': True,
        'clearing': True,
        'min_obstacle_height': 0.08,      # ignore the floor (+ small depth noise)
        'max_obstacle_height': 0.8,
        'obstacle_max_range': 4.0, 'obstacle_min_range': 0.0,
        'raytrace_max_range': 5.0, 'raytrace_min_range': 0.0,
        'expected_update_rate': 0.0,
        'inf_is_valid': False,
    },
}
inflation = {
    'plugin': 'nav2_costmap_2d::InflationLayer',
    'inflation_radius': 0.55,
    'cost_scaling_factor': 5.0,
}

# local costmap: 6x6 m around the robot (camera range ~4 m)
lc = p['local_costmap']['local_costmap']['ros__parameters']
lc.update({
    'global_frame': 'odom', 'robot_base_frame': 'base_link',
    'rolling_window': True, 'width': 6, 'height': 6, 'resolution': 0.05,
    'update_frequency': 5.0, 'publish_frequency': 2.0,
    'robot_radius': a.robot_radius,
    'plugins': ['voxel_layer', 'inflation_layer'],
    'voxel_layer': copy.deepcopy(voxel), 'inflation_layer': copy.deepcopy(inflation),
})
lc.pop('footprint', None)

# global costmap: no map, 24x24 m rolling window in odom
gc = p['global_costmap']['global_costmap']['ros__parameters']
gc.update({
    'global_frame': 'odom', 'robot_base_frame': 'base_link',
    'rolling_window': True, 'width': 24, 'height': 24, 'resolution': 0.05,
    'update_frequency': 2.0, 'publish_frequency': 1.0,
    'track_unknown_space': True,
    'robot_radius': a.robot_radius,
    'plugins': ['voxel_layer', 'inflation_layer'],
    'voxel_layer': copy.deepcopy(voxel), 'inflation_layer': copy.deepcopy(inflation),
})
gc.pop('footprint', None)

# ------------------------------------------------------------------ planner
pl = p['planner_server']['ros__parameters']
pl['GridBased'].update({'tolerance': 0.3, 'allow_unknown': True})   # plan through unseen space

# ------------------------------------------------------------------ controller (DWB)
cs = p['controller_server']['ros__parameters']
cs['controller_frequency'] = 10.0
cs['progress_checker'].update({'required_movement_radius': 0.3,
                               'movement_time_allowance': 15.0})
cs['general_goal_checker'].update({'xy_goal_tolerance': 0.25, 'yaw_goal_tolerance': 3.14})
dwb = cs['FollowPath']
dwb.update({
    'max_vel_x': 0.3, 'min_vel_x': 0.0, 'max_speed_xy': 0.3,
    'max_vel_theta': 1.0, 'acc_lim_theta': 2.5,
    'sim_time': 1.5, 'vx_samples': 20, 'vtheta_samples': 20,
    'BaseObstacle.scale': 0.05,      # keep a bit more distance from obstacles
    'PathAlign.scale': 24.0, 'GoalAlign.scale': 18.0,
    'PathDist.scale': 32.0, 'GoalDist.scale': 24.0,
})

# ------------------------------------------------------------------ odom frame everywhere
bt = p['bt_navigator']['ros__parameters']
bt['global_frame'] = 'odom'
bt['robot_base_frame'] = 'base_link'
bt['default_nav_to_pose_bt_xml'] = os.path.join(HERE, 'camera_bt.xml')

beh = p['behavior_server']['ros__parameters']
beh['global_frame'] = 'odom'
beh['robot_base_frame'] = 'base_link'
beh['costmap_topic'] = 'local_costmap/costmap_raw'

# collision monitor / velocity smoother defaults are fine; make sure velocity smoother
# limits match the controller
if 'velocity_smoother' in p:
    vs = p['velocity_smoother']['ros__parameters']
    vs['max_velocity'] = [0.3, 0.0, 1.0]
    vs['min_velocity'] = [-0.3, 0.0, -1.0]

# sim time for every node
for node in p.values():
    if isinstance(node, dict) and 'ros__parameters' in node:
        node['ros__parameters']['use_sim_time'] = True
    elif isinstance(node, dict):
        for sub in node.values():
            if isinstance(sub, dict) and 'ros__parameters' in sub:
                sub['ros__parameters']['use_sim_time'] = True



class NoAliasDumper(yaml.SafeDumper):
    # ROS 2 cannot parse YAML anchors/aliases (&id001 / *id001)
    def ignore_aliases(self, data):
        return True


yaml.dump(p, open(a.out, 'w'), Dumper=NoAliasDumper, sort_keys=False)
print(f'wrote {a.out}  (cloud topic: {a.cloud})')
