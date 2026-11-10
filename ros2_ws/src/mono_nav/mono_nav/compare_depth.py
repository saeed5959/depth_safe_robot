import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

class Compare(Node):
    def __init__(self):
        super().__init__('compare_depth')
        self.b = CvBridge()
        gt = Subscriber(self, Image, '/camera/depth_image')
        pr = Subscriber(self, Image, '/mono/depth')
        self.sync = ApproximateTimeSynchronizer([gt, pr], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.cb)

    def cb(self, g_msg, p_msg):
        g = self.b.imgmsg_to_cv2(g_msg)
        p = self.b.imgmsg_to_cv2(p_msg)
        m = np.isfinite(g) & (g > 0.1) & (g < 10.0)
        if m.sum() == 0:
            return
        absrel = np.mean(np.abs(p[m] - g[m]) / g[m])
        ratio = np.median(p[m] / g[m])
        self.get_logger().info(f'AbsRel={absrel:.3f}   median pred/gt={ratio:.2f}')

def main():
    rclpy.init()
    rclpy.spin(Compare())
