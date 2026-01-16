import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from percepcion.pid import pid
import math
import tf_transformations as tf
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data


class DarVueltas(Node):
    def __init__(self):
        super().__init__('dar_vueltas')
        self.subscriber_ = self.create_subscription(Int32, '/num_vueltas', self.callback, 10)
        self.subs_imu = self.create_subscription(Odometry, '/odom', self.imu_update, qos_profile_sensor_data)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.controlador = pid(0.045, 0.02, 0.0, 0)
        self.controlador.set_max_val(3)

        self.first_iteration = True
        self.rotation = 0.0          # grados acumulados
        self.yaw = 0.0               # rad
        self.num_vueltas = 0
        self.prev_angle = 0.0
        self.setpoint = 0.0

        self.get_logger().info('Dar vueltas node started')

    def callback(self, msg):
        if self.num_vueltas == 0 and msg.data != 0:
            self.num_vueltas = msg.data
            self.rotation = 0.0
            self.first_iteration = True

            self.setpoint = self.rotation + self.num_vueltas * 360.0  # <-- 360, no 356.25
            self.controlador.set_setpoint(self.setpoint)

            self.get_logger().info(f'Dando {msg.data} vueltas. Setpoint: {self.setpoint:.2f} deg')
            self.timer = self.create_timer(0.02, self.timer_callback)

    def timer_callback(self):
        value = self.controlador.update(self.rotation, 0.02)

        twist_msg = Twist()
        twist_msg.angular.z = float(value)
        twist_msg.linear.x = 0.0

        if abs(self.controlador.get_error()) < 0.5 and abs(value) < 0.5:
            self.get_logger().info('Deteniendo robot')
            self.get_logger().info(f'Yaw final: {math.degrees(self.yaw):.2f} deg')
            self.get_logger().info(f'Rotacion acumulada: {self.rotation:.2f} deg')

            twist_msg.angular.z = 0.0
            self.pub.publish(twist_msg)

            self.timer.cancel()
            self.controlador.reset()
            self.num_vueltas = 0
            self.rotation = 0.0
            self.first_iteration = True
            return

        self.get_logger().info(f'Rotacion: {self.rotation:.2f} deg | Yaw: {math.degrees(self.yaw):.2f} deg | Error: {self.controlador.get_error():.2f}')
        self.pub.publish(twist_msg)

    def imu_update(self, msg):
        q = msg.pose.pose.orientation
        orientation_list = [q.x, q.y, q.z, q.w]
        (_, _, self.yaw) = tf.euler_from_quaternion(orientation_list)

        if self.first_iteration:
            self.prev_angle = self.yaw
            self.first_iteration = False
            return

        delta_yaw = self.yaw - self.prev_angle
        if delta_yaw > math.pi:
            delta_yaw -= 2 * math.pi
        elif delta_yaw < -math.pi:
            delta_yaw += 2 * math.pi

        self.rotation -= math.degrees(delta_yaw)
        self.prev_angle = self.yaw


def main(args=None):
    rclpy.init(args=args)
    node = DarVueltas()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
