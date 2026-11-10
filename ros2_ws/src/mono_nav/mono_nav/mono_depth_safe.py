"""Depth Anything V2 + uncertainty head. Publishes the depth Nav2 should use on /mono/depth.

mode:
  raw     d_DA                               (same as mono_depth)
  margin  d_DA * exp(-c)                     (constant safety margin, c calibrated on val)
  mu      d_DA * exp(mu)                     (corrected depth, no uncertainty)
  safe    d_DA * exp(mu - k*sigma)           (our method, k calibrated on val)

k and c default to the values stored in the head checkpoint by train_head.py.

Floor snapping (floor_snap:=true, applied identically in every mode):
  Shrinking depth along a camera ray also LIFTS floor points (the camera looks down),
  so a conservative depth turns the floor into phantom obstacles and the robot gets stuck.
  Each pixel's ray hits the floor at depth d_floor. With the method's best estimate
  d_est (d_DA for raw/margin, d_DA*exp(mu) for mu/safe) the pixel is floor if
      log d_est - log d_floor >= -tau      (estimate at/behind the floor, or close to it)
  tau = floor_tau for raw/margin/mu; for safe tau = max(floor_tau, k*sigma), i.e. the
  pixel is floor if the floor lies inside its own uncertainty band. (A height threshold
  does not work: under-estimating floor depth by e lifts the point by H*e at ANY range.)
  Floor pixels get exactly d_floor.
  Floor pixels then still clear the costmap but never mark obstacles. Obstacle pixels
  keep the method's depth (raw / margin / mu / safe). Flat-floor assumption.
Debug topics (publish_debug:=true): /mono/depth_raw, /mono/depth_corrected, /mono/sigma.

ros2 run mono_nav mono_depth_safe --ros-args -p repo_path:=$DA \
    -p checkpoint:=$DA/checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
    -p head_ckpt:=/path/runs/head_v1/head_best.pt -p mode:=safe
"""
import time

import numpy as np
import torch
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from mono_nav.uncertainty_head import (DA2Features, UncertaintyHead, load_head, net_size,
                                       preprocess, upsample)

MODES = ('raw', 'margin', 'mu', 'safe')


class MonoDepthSafe(Node):
    def __init__(self):
        super().__init__('mono_depth')        # same node name as mono_depth (only one runs)
        for name, default in [('repo_path', ''), ('checkpoint', ''), ('encoder', 'vits'),
                              ('max_depth', 20.0), ('input_size', 518), ('head_ckpt', ''),
                              ('mode', 'safe'), ('k', -1.0), ('margin_c', -1.0),
                              ('fp16', True), ('publish_debug', False),
                              ('out_topic', '/mono/depth'), ('floor_snap', True),
                              ('floor_tau', 0.15), ('floor_frame', 'odom'),
                              ('floor_z', -0.05), ('out_scale', 1.0)]:
            self.declare_parameter(name, default)
        P = lambda n: self.get_parameter(n).value
        self.mode = P('mode')
        assert self.mode in MODES, f'mode must be one of {MODES}'
        self.input_size = int(P('input_size'))
        self.debug = bool(P('publish_debug'))
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.amp = bool(P('fp16')) and self.dev == 'cuda'
        torch.backends.cudnn.benchmark = True

        self.da2 = DA2Features(P('repo_path'), P('checkpoint'), P('encoder'),
                               float(P('max_depth'))).to(self.dev)
        self.head, self.k, self.c = None, 0.0, 0.0
        if self.mode != 'raw':
            if not P('head_ckpt'):
                raise RuntimeError(f'mode={self.mode} needs head_ckpt')
            self.head, ck = load_head(P('head_ckpt'), self.dev)
            if 'k' not in ck or 'margin_c' not in ck:
                self.get_logger().error(
                    'head checkpoint has NO calibration (k, margin_c): training did not finish. '
                    'Run: python3 train_head.py --calibrate_only --out <run dir>. '
                    'Falling back to k=2, c=0 (margin mode = raw depth!)')
            self.k = float(P('k')) if P('k') >= 0 else float(ck.get('k', 2.0))
            self.c = float(P('margin_c')) if P('margin_c') >= 0 else float(ck.get('margin_c', 0.0))
            if int(ck.get('input_size', self.input_size)) != self.input_size:
                self.get_logger().warn(
                    f"head was trained with input_size={ck['input_size']} but node uses "
                    f"{self.input_size}; switching to {ck['input_size']}")
                self.input_size = int(ck['input_size'])
        self.size = None
        self.floor_snap = bool(P('floor_snap'))
        self.floor_tau = float(P('floor_tau'))
        self.floor_frame = P('floor_frame')
        self.floor_z = float(P('floor_z'))   # height of the real floor in floor_frame
        self.K = None
        self.A = None          # per-pixel vertical component of the (z=1) camera ray, floor frame
        self.H = None          # camera height above the floor
        if self.floor_snap:
            self.tf_buf = Buffer()
            self.tf_lis = TransformListener(self.tf_buf, self)
            self.create_subscription(CameraInfo, '/camera/camera_info', self.info_cb, 1)
        # out_scale < 1 publishes a smaller depth image (and a matching camera_info on
        # /mono/camera_info): far fewer points for depth_image_proc and the costmaps.
        self.out_scale = float(P('out_scale'))
        self.info_msg = None
        self.pub_info = self.create_publisher(CameraInfo, '/mono/camera_info', 1)
        if not self.floor_snap:
            self.create_subscription(CameraInfo, '/camera/camera_info', self.info_cb, 1)
        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/image', self.cb, 1)
        self.pub = self.create_publisher(Image, P('out_topic'), 1)
        if self.debug:
            self.pub_raw = self.create_publisher(Image, '/mono/depth_raw', 1)
            self.pub_cor = self.create_publisher(Image, '/mono/depth_corrected', 1)
            self.pub_sig = self.create_publisher(Image, '/mono/sigma', 1)
        self.get_logger().info(f'mode={self.mode} k={self.k:.2f} c={self.c:.3f} '
                               f'input_size={self.input_size} device={self.dev} fp16={self.amp} '
                               f'floor_snap={self.floor_snap}')

    def info_cb(self, m):
        self.info_msg = m
        if self.K is None:
            self.K = (m.k[0], m.k[4], m.k[2], m.k[5], m.width, m.height)
            self.info_msg = m

    def _scaled_info(self, header, h, w):
        m = CameraInfo()
        src = self.info_msg
        m.header = header
        m.width, m.height = w, h
        sx, sy = w / src.width, h / src.height
        k = list(src.k)
        k[0] *= sx; k[2] *= sx; k[4] *= sy; k[5] *= sy
        m.k = k
        p_ = list(src.p)
        p_[0] *= sx; p_[2] *= sx; p_[5] *= sy; p_[6] *= sy
        m.p = p_
        m.r = src.r
        m.d = src.d
        m.distortion_model = src.distortion_model
        return m

    def _floor_geometry(self, frame_id, h, w):
        """Camera height and per-pixel ray slope w.r.t. the floor (computed once).
        Height/pitch of the camera above a flat floor do not change while driving."""
        if self.K is None:
            return False
        try:
            t = self.tf_buf.lookup_transform(self.floor_frame, frame_id, rclpy.time.Time())
        except Exception as e:                       # TF not there yet
            self.get_logger().info(f'waiting for TF {self.floor_frame}->{frame_id}: {e}',
                                   throttle_duration_sec=2.0)
            return False
        q = t.transform.rotation
        x, y, z, ww = q.x, q.y, q.z, q.w
        r20 = 2 * (x * z - ww * y)                    # third row of the rotation matrix
        r21 = 2 * (y * z + ww * x)
        r22 = 1 - 2 * (x * x + y * y)
        fx, fy, cx, cy, kw, kh = self.K
        sx, sy = w / kw, h / kh                       # camera_info may differ in size
        u = (torch.arange(w, device=self.dev, dtype=torch.float32) + 0.5) / sx
        v = (torch.arange(h, device=self.dev, dtype=torch.float32) + 0.5) / sy
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        # ray (x, y, 1) in the optical frame -> its z component in the floor frame
        self.A = r20 * (uu - cx) / fx + r21 * (vv - cy) / fy + r22
        self.H = float(t.transform.translation.z) - self.floor_z
        self.get_logger().info(
            f'floor snapping ready: camera z={t.transform.translation.z:.3f} in {self.floor_frame}, '
            f'floor at z={self.floor_z:.3f} -> camera {self.H:.3f} m above floor')
        return True

    def _snap(self, out, est, tau):
        """Replace floor pixels by the exact floor depth. tau: scalar or per-pixel tensor."""
        A = self.A
        down = A < -1e-3
        d_floor = -self.H / A.clamp(max=-1e-3)
        floor = down & (torch.log(est.clamp(min=1e-3)) - torch.log(d_floor) >= -tau)
        return torch.where(floor, d_floor, out)

    def _img(self, arr, header):
        m = self.bridge.cv2_to_imgmsg(np.ascontiguousarray(arr, dtype=np.float32), encoding='32FC1')
        m.header = header
        return m

    @torch.inference_mode()
    def cb(self, msg):
        t0 = time.time()
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h0, w0 = bgr.shape[:2]
        h, w = max(1, round(h0 * self.out_scale)), max(1, round(w0 * self.out_scale))
        if self.size is None:
            self.size = net_size(h0, w0, self.input_size)
        rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).to(self.dev)
        rgb = rgb.permute(2, 0, 1)[None]
        x = preprocess(rgb, self.size)
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
            d_da, feat = self.da2(x)
        d_da = d_da.float()
        est = d_da                                    # best estimate, used to find the floor
        tau = self.floor_tau
        if self.head is None:
            out = d_da
        else:
            mu, s = self.head(UncertaintyHead.make_input(feat, d_da, x))
            mu = upsample(mu, d_da.shape[-2:])
            s = upsample(s, d_da.shape[-2:])
            if self.mode == 'margin':
                out = d_da * float(np.exp(-self.c))
            elif self.mode == 'mu':
                out = d_da * torch.exp(mu) * 1.3
                est = out
            else:
                sig = torch.exp(s)
                out = d_da * torch.exp(mu - self.k * sig)
                est = d_da * torch.exp(mu)
                tau = (self.k * sig).clamp(min=self.floor_tau)
        out = upsample(out, (h, w))[0]
        if self.floor_snap:
            if self.A is None and not self._floor_geometry(msg.header.frame_id, h, w):
                return                                # do not publish unsnapped depth
            if torch.is_tensor(tau):
                tau = upsample(tau, (h, w))[0]
            out = self._snap(out, upsample(est, (h, w))[0], tau)
        out = out.cpu().numpy()
        # camera_info with EXACTLY this depth image's stamp (published first). depth_image_proc
        # pairs depth + camera_info by exact stamp with a short queue; DA2 output arrives
        # ~1 s late, so the original /camera/camera_info has long left that queue.
        if self.info_msg is not None:
            self.pub_info.publish(self._scaled_info(msg.header, h, w))
        self.pub.publish(self._img(out, msg.header))
        if self.debug and self.head is not None:
            self.pub_raw.publish(self._img(upsample(d_da, (h, w))[0].cpu().numpy(), msg.header))
            self.pub_cor.publish(self._img(upsample(d_da * torch.exp(mu), (h, w))[0].cpu().numpy(),
                                           msg.header))
            self.pub_sig.publish(self._img(upsample(torch.exp(s), (h, w))[0].cpu().numpy(),
                                           msg.header))
        self.get_logger().info(f'total {1e3 * (time.time() - t0):.0f} ms  (out {w}x{h})',
                               throttle_duration_sec=5.0)


def main():
    rclpy.init()
    node = MonoDepthSafe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
