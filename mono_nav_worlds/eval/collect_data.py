#!/usr/bin/env python3
"""Record (RGB, GT depth) pairs for training the uncertainty head.

Runs next to episode_runner.py. Saves only while an episode is active (/eval/episode),
and only when the robot moved >= min_dist or turned >= min_turn since the last frame,
so there are no near-duplicates while standing still.

Layout:
  <out>/<world>/<method>/rgb/000123.png       8-bit RGB
  <out>/<world>/<method>/depth/000123.png     uint16, millimeters, 0 = invalid / inf
  <out>/<world>/<method>/index.jsonl          one line per frame (episode, stamp, pose)
  <out>/<world>/<method>/camera_info.json

Depth Anything is frozen, so its prediction is NOT saved: recompute it offline from rgb.
Join index.jsonl with results.csv on (method, episode) to label frames by outcome.

usage (normally started by eval_stack.launch.py collect:=true):
  python3 collect_data.py --out dataset --method gt
"""
import argparse
import json
import math
import os
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Collector(Node):
    def __init__(self, a):
        super().__init__('collect_data', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.a = a
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.ep = {'active': False}
        self.odom = None
        self.last = None
        self.n = 0
        self.dir = None
        self.cam_info = None

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, '/eval/episode', self.ep_cb, latched)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 20)
        self.create_subscription(CameraInfo, a.info_topic, self.info_cb, 1)
        rgb = Subscriber(self, Image, a.rgb_topic)
        dep = Subscriber(self, Image, a.depth_topic)
        self.sync = ApproximateTimeSynchronizer([rgb, dep], queue_size=10, slop=0.02)
        self.sync.registerCallback(self.pair_cb)
        self.get_logger().info(f'collecting into {a.out}  (method={a.method})')

    def ep_cb(self, m):
        info = json.loads(m.data)
        with self.lock:
            self.ep = info
            if info.get('active'):
                self.last = None
                self.dir = os.path.join(self.a.out, info['world'], self.a.method)
                for sub in ('rgb', 'depth'):
                    os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
                idx = os.path.join(self.dir, 'index.jsonl')
                if self.n == 0 and os.path.exists(idx):          # resume numbering
                    self.n = sum(1 for _ in open(idx))
                if self.cam_info and not os.path.exists(os.path.join(self.dir, 'camera_info.json')):
                    json.dump(self.cam_info, open(os.path.join(self.dir, 'camera_info.json'), 'w'))

    def odom_cb(self, m):
        p = m.pose.pose
        with self.lock:
            self.odom = (p.position.x, p.position.y, yaw_of(p.orientation))

    def info_cb(self, m):
        if self.cam_info is None:
            self.cam_info = {'width': m.width, 'height': m.height, 'K': list(m.k),
                             'frame_id': m.header.frame_id}

    def pair_cb(self, rgb_msg, depth_msg):
        with self.lock:
            ep, od = dict(self.ep), self.odom
        if not ep.get('active') or od is None or self.dir is None:
            return
        if self.last is not None:
            moved = math.hypot(od[0] - self.last[0], od[1] - self.last[1])
            turned = abs(math.atan2(math.sin(od[2] - self.last[2]), math.cos(od[2] - self.last[2])))
            if moved < self.a.min_dist and turned < math.radians(self.a.min_turn):
                return
        self.last = od

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        d = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough').astype(np.float32)
        d[~np.isfinite(d)] = 0.0
        d_mm = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
        name = f'{self.n:06d}.png'
        cv2.imwrite(os.path.join(self.dir, 'rgb', name), rgb)
        cv2.imwrite(os.path.join(self.dir, 'depth', name), d_mm)
        stamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
        rec = {'frame': self.n, 'file': name, 'world': ep['world'], 'method': self.a.method,
               'episode': ep['episode'], 'stamp': round(stamp, 3),
               'odom': [round(v, 4) for v in od],
               'valid_frac': round(float((d_mm > 0).mean()), 3)}
        with open(os.path.join(self.dir, 'index.jsonl'), 'a') as f:
            f.write(json.dumps(rec) + '\n')
        self.n += 1
        if self.n % 100 == 0:
            self.get_logger().info(f'{self.n} frames ({ep["world"]}/{self.a.method})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='dataset')
    ap.add_argument('--method', required=True, help='which stack drove the robot (gt | mono)')
    ap.add_argument('--rgb_topic', default='/camera/image')
    ap.add_argument('--depth_topic', default='/camera/depth_image')
    ap.add_argument('--info_topic', default='/camera/camera_info')
    ap.add_argument('--min_dist', type=float, default=0.20)
    ap.add_argument('--min_turn', type=float, default=10.0, help='degrees')
    a, _ = ap.parse_known_args()          # ignore --ros-args added by launch
    rclpy.init()
    rclpy.spin(Collector(a))


if __name__ == '__main__':
    main()
