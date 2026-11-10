import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomToTf(Node):
    """Publish the odom -> base_link TF from the ground-truth /odom topic."""

    def __init__(self):
        super().__init__('odom_to_tf')
        self.br = TransformBroadcaster(self)
        self.create_subscription(Odometry, '/odom', self.cb, 10)

    def cb(self, m):
        t = TransformStamped()
        t.header.stamp = m.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        p = m.pose.pose.position
        t.transform.translation.x = p.x
        t.transform.translation.y = p.y
        t.transform.translation.z = p.z
        t.transform.rotation = m.pose.pose.orientation
        self.br.sendTransform(t)


def main():
    rclpy.init()
    rclpy.spin(OdomToTf())
