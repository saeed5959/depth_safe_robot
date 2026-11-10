#!/usr/bin/env python3
"""Run a list of navigation episodes against an already running sim + Nav2 stack.

Per episode:
  cancel nav -> stop robot -> teleport to start (ign service set_pose) -> clear costmaps
  -> wait for fresh sensor data -> goToPose(goal) -> monitor until a terminal outcome:

  success    Nav2 SUCCEEDED and robot within goal_tol of the goal, no collision
  collision  contact sensor (/bumper) touched anything but the floor  (ends the run)
  timeout    sim time > t_max  (2.5 * A* length / v_max + 20 s)
  stuck      moved < 0.10 m in the last --stuck_time s
  nav_failed Nav2 returned FAILED / CANCELED (e.g. no path, recoveries exhausted)

Results are appended to <out>/results.csv (resume-safe: finished episodes are skipped),
trajectories to <out>/traj/<method>/<episode>.npz.

usage (stack already running for this world + method):
  python3 episode_runner.py --episodes episodes/test_02.yaml --method gt --out results \
      --worlds_dir ~/depth_safe_robot/mono_nav_worlds/worlds
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np
import yaml
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_map import WorldMap  # noqa: E402

try:
    from ros_gz_interfaces.msg import Contacts
except ImportError:
    Contacts = None

FIELDS = ['method', 'world', 'episode', 'pair', 'repeat', 'kind', 'difficulty', 'tags',
          'outcome', 'success', 'time_s', 't_max', 'astar_len', 'path_len', 'spl',
          'min_clearance', 'n_recoveries', 'final_dist', 'collided_with', 'wall_time_s']


# ---------------------------------------------------------------------- 2D pose math
def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


# ---------------------------------------------------------------------- monitor node
class Monitor(Node):
    """Odometry, bumper contacts, cmd_vel, episode-state topic. Spun in its own thread."""

    def __init__(self, bumper_topic, ignore):
        super().__init__('eval_monitor', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.lock = threading.Lock()
        self.odom = None          # (t, x, y, yaw) in odom frame
        self.ignore = ignore
        self.contacts = []        # (t, name)
        self.recording = False
        self.traj = []
        self.create_subscription(Odometry, '/odom', self.odom_cb, 20)
        self.bumper_topic = bumper_topic
        if Contacts is not None:
            self.create_subscription(Contacts, bumper_topic, self.contact_cb, 50)
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.ep_pub = self.create_publisher(String, '/eval/episode', latched)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, m):
        p = m.pose.pose
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        with self.lock:
            self.odom = (t, p.position.x, p.position.y, yaw_of(p.orientation))
            if self.recording and (not self.traj or t - self.traj[-1][0] >= 0.1):
                self.traj.append(self.odom)

    def contact_cb(self, msg):
        t = self.now()
        for c in msg.contacts:
            names = [c.collision1.name, c.collision2.name]
            others = [n for n in names if 'mono_bot' not in n]
            if not others:
                continue
            o = others[0]
            if any(k in o.lower() for k in self.ignore):
                continue
            with self.lock:
                self.contacts.append((t, o))

    def get_odom(self):
        with self.lock:
            return self.odom

    def start_recording(self):
        with self.lock:
            self.traj, self.contacts, self.recording = [], [], True

    def stop_recording(self):
        with self.lock:
            self.recording = False
            return list(self.traj)

    def contacts_after(self, t0):
        with self.lock:
            return [c for c in self.contacts if c[0] >= t0]

    def stop_robot(self, n=10):
        for _ in range(n):
            self.cmd.publish(Twist())
            time.sleep(0.02)

    def publish_episode(self, info):
        self.ep_pub.publish(String(data=json.dumps(info)))


# ---------------------------------------------------------------------- runner
class Runner:
    def __init__(self, a):
        self.a = a
        self.cfg = yaml.safe_load(open(a.episodes))
        self.world = self.cfg['world']
        wd = os.path.expanduser(a.worlds_dir)
        self.wm = WorldMap(os.path.join(wd, f'{self.world}.sdf'),
                           os.path.join(wd, f'{self.world}_goals.yaml'))
        sp = self.cfg['spawn']
        self.spawn = (sp['x'], sp['y'], sp['yaw'])

        rclpy.init()
        self.mon = Monitor(a.bumper_topic, [s.strip().lower() for s in a.ignore_contacts.split(',')])
        self.exe = SingleThreadedExecutor()
        self.exe.add_node(self.mon)
        threading.Thread(target=self.exe.spin, daemon=True).start()

        self.nav = BasicNavigator()
        self.nav.set_parameters([Parameter('use_sim_time', value=True)])

        os.makedirs(os.path.join(a.out, 'traj', a.method), exist_ok=True)
        self.csv_path = os.path.join(a.out, 'results.csv')
        self.done = set()
        if os.path.exists(self.csv_path):
            for r in csv.DictReader(open(self.csv_path)):
                self.done.add((r['method'], r['episode']))

    # -------------------------------------------------------------- helpers
    def log(self, *x):
        print(f'[runner {self.world}/{self.a.method}]', *x, flush=True)

    def sim_sleep(self, dt, wall_limit=None):
        t0, w0 = self.mon.now(), time.time()
        wall_limit = wall_limit or max(10.0, 20 * dt)
        while self.mon.now() - t0 < dt and time.time() - w0 < wall_limit:
            time.sleep(0.02)

    def ign(self, service, reqtype, reptype, req):
        cmd = ['ign', 'service', '-s', service, '--reqtype', reqtype, '--reptype', reptype,
               '--timeout', '3000', '--req', req]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return 'true' in r.stdout

    def teleport(self, x, y, yaw):
        req = (f'name: "{self.a.robot}", position: {{x: {x}, y: {y}, z: {self.a.spawn_z}}}, '
               f'orientation: {{x: 0, y: 0, z: {math.sin(yaw / 2)}, w: {math.cos(yaw / 2)}}}')
        return self.ign(f'/world/{self.world}/set_pose', 'ignition.msgs.Pose',
                        'ignition.msgs.Boolean', req)

    def world_pose(self):
        o = self.mon.get_odom()
        return None if o is None else compose(self.T_wo, o[1:])

    def wait_odom(self):
        w0 = time.time()
        while self.mon.get_odom() is None:
            if time.time() - w0 > 60:
                raise RuntimeError('no /odom - is the stack running?')
            time.sleep(0.1)

    def calibrate(self):
        """T_world_odom from a known world pose (robot teleported to spawn)."""
        self.wait_odom()
        if not self.teleport(*self.spawn):
            self.log('WARNING: teleport to spawn failed; assuming robot still at spawn')
        self.sim_sleep(1.0)
        o = self.mon.get_odom()[1:]
        self.T_wo = compose(self.spawn, inverse(o))
        self.log(f'odom->world offset: x={self.T_wo[0]:.3f} y={self.T_wo[1]:.3f} '
                 f'yaw={math.degrees(self.T_wo[2]):.1f} deg')

    def set_rtf(self):
        if self.a.rtf > 0:
            ok = self.ign(f'/world/{self.world}/set_physics', 'ignition.msgs.Physics',
                          'ignition.msgs.Boolean', f'real_time_factor: {self.a.rtf}')
            self.log(f'real_time_factor={self.a.rtf} ({"ok" if ok else "FAILED"})')

    def goal_msg(self, gx, gy, gyaw):
        x, y, yaw = compose(inverse(self.T_wo), (gx, gy, gyaw))
        p = PoseStamped()
        p.header.frame_id = 'odom'
        p.pose.position.x, p.pose.position.y = x, y
        p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return p

    # -------------------------------------------------------------- one episode
    def run_episode(self, ep):
        a = self.a
        s, g = ep['start'], ep['goal']
        if not self.nav.isTaskComplete():
            self.nav.cancelTask()
        self.mon.stop_robot()

        ok = False
        for _ in range(3):
            if self.teleport(s['x'], s['y'], s['yaw']):
                self.sim_sleep(1.0)
                wp = self.world_pose()
                if wp and math.hypot(wp[0] - s['x'], wp[1] - s['y']) < 0.15:
                    ok = True
                    break
        if not ok:
            self.log(f'{ep["id"]}: teleport failed, skipping')
            return None
        self.mon.stop_robot()
        self.nav.clearAllCostmaps()
        self.sim_sleep(a.settle)                     # let fresh depth refill the costmaps

        info = {'active': True, 'world': self.world, 'method': a.method, 'episode': ep['id']}
        self.mon.publish_episode(info)
        self.mon.start_recording()
        t0, w0 = self.mon.now(), time.time()
        gyaw = math.atan2(g['y'] - s['y'], g['x'] - s['x'])
        accepted = self.nav.goToPose(self.goal_msg(g['x'], g['y'], gyaw))

        outcome, collided, n_rec = (None if accepted is not False else 'nav_failed'), '', 0
        min_clear = math.inf
        last_prog = (t0, (s['x'], s['y']))
        while outcome is None:
            t = self.mon.now() - t0
            wp = self.world_pose()
            if wp is not None:
                c = self.wm.clearance(wp[0], wp[1])
                min_clear = min(min_clear, c)
                if math.hypot(wp[0] - last_prog[1][0], wp[1] - last_prog[1][1]) > 0.10:
                    last_prog = (self.mon.now(), wp[:2])
                if a.collision in ('geometric', 'both') and c < a.body_radius:
                    outcome, collided = 'collision', 'geometric'
                    break
            hits = self.mon.contacts_after(t0 + a.contact_grace)
            if a.collision in ('contact', 'both') and hits:
                outcome, collided = 'collision', hits[0][1]
                break
            if t > ep['t_max']:
                outcome = 'timeout'
                break
            if self.mon.now() - last_prog[0] > a.stuck_time:
                outcome = 'stuck'
                break
            if self.nav.isTaskComplete():            # spins the navigator ~0.1 s
                res = self.nav.getResult()
                wp = self.world_pose() or (s['x'], s['y'], 0.0)
                d = math.hypot(wp[0] - g['x'], wp[1] - g['y'])
                if res == TaskResult.SUCCEEDED and d <= a.goal_tol:
                    outcome = 'success'
                else:
                    outcome = 'nav_failed'
                break
            fb = self.nav.getFeedback()
            if fb is not None:
                n_rec = max(n_rec, fb.number_of_recoveries)

        if not self.nav.isTaskComplete():
            self.nav.cancelTask()
        self.mon.stop_robot()
        self.mon.publish_episode({'active': False, 'world': self.world, 'method': a.method,
                                  'episode': ep['id'], 'outcome': outcome})
        traj = self.mon.stop_recording()

        t_ep = self.mon.now() - t0
        wpath = np.array([compose(self.T_wo, p[1:]) for p in traj]) if traj else np.zeros((0, 3))
        plen = float(np.sum(np.hypot(*np.diff(wpath[:, :2], axis=0).T))) if len(wpath) > 1 else 0.0
        wp = self.world_pose() or (s['x'], s['y'], 0.0)
        fd = math.hypot(wp[0] - g['x'], wp[1] - g['y'])
        succ = int(outcome == 'success')
        spl = succ * ep['astar_len'] / max(ep['astar_len'], plen) if plen > 0 else 0.0
        np.savez_compressed(os.path.join(a.out, 'traj', a.method, f"{ep['id']}.npz"),
                            t=np.array([p[0] for p in traj]) - t0, pose=wpath,
                            start=[s['x'], s['y'], s['yaw']], goal=[g['x'], g['y']])
        return {
            'method': a.method, 'world': self.world, 'episode': ep['id'],
            'pair': ep.get('pair', ep['id']), 'repeat': ep.get('repeat', 0),
            'kind': ep.get('kind', ''), 'difficulty': ep['difficulty'],
            'tags': '|'.join(ep['tags']), 'outcome': outcome, 'success': succ,
            'time_s': round(t_ep, 2), 't_max': ep['t_max'], 'astar_len': ep['astar_len'],
            'path_len': round(plen, 2), 'spl': round(spl, 3),
            'min_clearance': round(min_clear, 3) if min_clear < math.inf else '',
            'n_recoveries': n_rec, 'final_dist': round(fd, 2), 'collided_with': collided,
            'wall_time_s': round(time.time() - w0, 1),
        }

    # -------------------------------------------------------------- main loop
    def run(self):
        a = self.a
        self.log('waiting for Nav2 to become active ...')
        t_wait = time.time()
        for node in ('controller_server', 'bt_navigator'):
            cli = f'{node}/get_state'
            while not self.nav.get_service_names_and_types() or \
                    not any(n == f'/{cli}' for n, _ in self.nav.get_service_names_and_types()):
                if time.time() - t_wait > a.nav_timeout:
                    self.log(f'ERROR: {node} not up after {a.nav_timeout:.0f} s. '
                             f'Is the stack running? Check the *_stack.log file.')
                    sys.exit(2)
                time.sleep(1.0)
        self.nav.waitUntilNav2Active(localizer='controller_server')
        if a.collision != 'geometric' and (Contacts is None or
                                           self.mon.count_publishers(a.bumper_topic) == 0):
            self.log(f'WARNING: no publisher on {a.bumper_topic} (or ros_gz_interfaces missing) '
                     f'-> falling back to geometric collision check')
            a.collision = 'geometric'
        self.set_rtf()
        self.calibrate()

        eps = self.cfg['episodes']
        if a.only:
            eps = [e for e in eps if any(k in e['id'] for k in a.only.split(','))]
        if a.limit:
            eps = eps[:a.limit]
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            for i, ep in enumerate(eps):
                if (a.method, ep['id']) in self.done:
                    continue
                row = self.run_episode(ep)
                if row is None:
                    continue
                w.writerow(row)
                f.flush()
                self.log(f"[{i + 1}/{len(eps)}] {ep['id']:<28} {row['difficulty']:<6} "
                         f"{row['outcome']:<10} t={row['time_s']:6.1f}s len={row['path_len']:5.1f} "
                         f"clear={row['min_clearance']} {row['collided_with']}")
        self.mon.publish_episode({'active': False, 'world': self.world, 'method': a.method,
                                  'episode': None, 'finished': True})
        self.log('done')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', required=True)
    ap.add_argument('--method', required=True, help='label: gt | mono | mono_margin | mono_sigma ...')
    ap.add_argument('--out', default='results')
    ap.add_argument('--worlds_dir', default='~/depth_safe_robot/mono_nav_worlds/worlds')
    ap.add_argument('--robot', default='mono_bot')
    ap.add_argument('--spawn_z', type=float, default=0.06)
    ap.add_argument('--goal_tol', type=float, default=0.30, help='Nav2 xy tolerance is 0.25')
    ap.add_argument('--collision', choices=['contact', 'geometric', 'both'], default='contact')
    ap.add_argument('--bumper_topic', default='/bumper')
    ap.add_argument('--ignore_contacts', default='ground,floor',
                    help='contact names containing these are ignored')
    ap.add_argument('--contact_grace', type=float, default=1.0, help='s after start')
    ap.add_argument('--body_radius', type=float, default=0.21,
                    help='geometric check: collision if obstacle closer than this to base_link')
    ap.add_argument('--settle', type=float, default=3.0, help='sim s to refill costmaps')
    ap.add_argument('--stuck_time', type=float, default=40.0)
    ap.add_argument('--nav_timeout', type=float, default=120.0, help='wall s to wait for Nav2 at start')
    ap.add_argument('--rtf', type=float, default=0.0, help='set Gazebo real_time_factor (0 = keep)')
    ap.add_argument('--only', default='', help='comma list of substrings of episode ids')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    r = Runner(a)
    try:
        r.run()
    finally:
        r.mon.stop_robot()
        r.exe.shutdown()
        r.nav.destroy_node()
        r.mon.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
