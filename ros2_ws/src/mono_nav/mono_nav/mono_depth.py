import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


class MonoDepth(Node):
    def __init__(self):
        super().__init__('mono_depth')
        self.declare_parameter('repo_path', '')
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('encoder', 'vits')
        self.declare_parameter('max_depth', 20.0)      # 20 for Hypersim (indoor)
        self.declare_parameter('input_size', 364)      # must be a multiple of 14
        self.declare_parameter('fp16', False)           # pure fp16 weights
        self.declare_parameter('compile', True)       # torch.compile (PyTorch >= 2.1)
        self.declare_parameter('full_res_output', True)  # False = publish at network resolution

        repo = self.get_parameter('repo_path').value
        ckpt = self.get_parameter('checkpoint').value
        enc = self.get_parameter('encoder').value
        self.input_size = int(self.get_parameter('input_size').value)
        self.fp16 = bool(self.get_parameter('fp16').value)
        self.full_res = bool(self.get_parameter('full_res_output').value)

        # import the model code from the cloned repo
        sys.path.insert(0, os.path.join(repo, 'metric_depth'))
        from depth_anything_v2.dpt import DepthAnythingV2

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.fp16 = False
        self.dtype = torch.float16 if self.fp16 else torch.float32
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        model = DepthAnythingV2(**MODEL_CONFIGS[enc],
                                max_depth=self.get_parameter('max_depth').value)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model = model.to(self.device, dtype=self.dtype).eval()
        if self.get_parameter('compile').value:
            self.model = torch.compile(self.model)

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device,
                                 dtype=self.dtype).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device,
                                dtype=self.dtype).view(1, 3, 1, 1)
        self.net_size = None      # set from the first frame
        self.warm = False

        self.get_logger().info(
            f'Depth Anything V2 ({enc}) loaded on {self.device}, '
            f'{"fp16" if self.fp16 else "fp32"}, input_size={self.input_size}')

        self.bridge = CvBridge()
        # queue size 1: always process the newest frame, drop old ones (no lag)
        self.create_subscription(Image, '/camera/image', self.cb, 1)
        self.pub = self.create_publisher(Image, '/mono/depth', 1)

    def _net_size(self, h, w):
        # short side -> input_size, both sides multiples of 14 (ViT patch size)
        s = self.input_size / min(h, w)
        return (max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14))

    def _sync(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()

    @torch.inference_mode()
    def _warmup(self):
        x = torch.zeros(1, 3, *self.net_size, device=self.device, dtype=self.dtype)
        for _ in range(3):
            self.model(x)
        self._sync()
        self.warm = True
        self.get_logger().info(f'warm-up done, network size {self.net_size}')

    @torch.inference_mode()
    def cb(self, msg):
        t0 = time.time()
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = bgr.shape[:2]
        if self.net_size is None:
            self.net_size = self._net_size(h, w)
        if not self.warm:
            self._warmup()
            t0 = time.time()

        # preprocessing on GPU: BGR->RGB, HWC->CHW, /255, resize, normalize
        x = torch.from_numpy(np.ascontiguousarray(bgr)).to(self.device, non_blocking=True)
        x = x.flip(-1).permute(2, 0, 1).unsqueeze(0).to(self.dtype).div_(255)
        x = F.interpolate(x, size=self.net_size, mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        self._sync(); t1 = time.time()

        d = self.model(x)                                   # (1, H', W') metric depth
        self._sync(); t2 = time.time()

        if self.full_res:
            d = F.interpolate(d[:, None], size=(h, w), mode='bilinear',
                              align_corners=False)[0, 0]
        else:
            d = d[0]
        depth = d.float().cpu().numpy()
        t3 = time.time()

        out = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        out.header = msg.header        # same timestamp + camera_optical_frame
        self.pub.publish(out)
        self.get_logger().info(
            f'pre {1e3*(t1-t0):.0f} | net {1e3*(t2-t1):.0f} | '
            f'post {1e3*(t3-t2):.0f} | total {1e3*(time.time()-t0):.0f} ms',
            throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = MonoDepth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
