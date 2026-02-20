import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from osr_perception_challenge.pid import pid
import math
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data

import threading

def yaw_from_quaternion(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

class DarVueltas(Node):
    IDLE = 0         # listo para aceptar /num_vueltas
    RUNNING = 1      # ejecutando vueltas; IGNORAR /num_vueltas
    WAIT_ENTER = 2   # terminó; IGNORAR /num_vueltas hasta Enter

    def __init__(self):
        super().__init__('dar_vueltas')

        self.subscriber_ = self.create_subscription(Int32, '/num_vueltas', self.callback, 10)
        self.subs_imu = self.create_subscription(Odometry, '/odom', self.imu_update, qos_profile_sensor_data)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.controlador = pid(0.045, 0.02, 0.0, 0)
        self.controlador.set_max_val(3)

        self.first_iteration = True
        self.rotation = 0.0
        self.yaw = 0.0
        self.num_vueltas = 0
        self.prev_angle = 0.0
        self.setpoint = 0.0

        self.timer = None
        self.state = self.IDLE

        self._lock = threading.Lock()
        self._enter_thread = None

        self.get_logger().info('Dar vueltas node started (IDLE: esperando /num_vueltas)')

    def callback(self, msg: Int32):
        with self._lock:
            if self.state != self.IDLE:
                # Ignorar mientras está corriendo o esperando Enter
                return

            n = int(msg.data)
            if n == 0:
                return

            self.num_vueltas = n
            self.rotation = 0.0
            self.first_iteration = True

            self.setpoint = self.num_vueltas * 360.0
            self.controlador.set_setpoint(self.setpoint)

            self.state = self.RUNNING
            self.get_logger().info(f'Dando {self.num_vueltas} vueltas. Setpoint: {self.setpoint:.2f} deg')

            if self.timer is not None:
                self.timer.cancel()
            self.timer = self.create_timer(0.02, self.timer_callback)

    def timer_callback(self):
        dt = 0.02
        value = self.controlador.update(self.rotation, dt)

        twist_msg = Twist()
        twist_msg.angular.z = float(value)
        twist_msg.linear.x = 0.0

        # condición de parada
        if abs(self.controlador.get_error()) < 0.5 and abs(value) < 0.5:
            twist_msg.angular.z = 0.0
            self.pub.publish(twist_msg)

            with self._lock:
                if self.timer is not None:
                    self.timer.cancel()
                    self.timer = None

                self.controlador.reset()
                self.num_vueltas = 0
                self.rotation = 0.0
                self.first_iteration = True

                # Pasamos a WAIT_ENTER: no aceptar /num_vueltas hasta Enter
                self.state = self.WAIT_ENTER

            self.get_logger().info('Vueltas terminadas. Pulsa Enter para aceptar un nuevo /num_vueltas.')

            # Lanzar el thread de espera (solo si no existe ya)
            self._start_wait_enter_thread()
            return

        self.pub.publish(twist_msg)

    def _start_wait_enter_thread(self):
        if self._enter_thread is not None and self._enter_thread.is_alive():
            return

        def wait_enter():
            try:
                input()  # Espera Enter en terminal
            except EOFError:
                # Si no hay stdin (ej. launch sin terminal), no podemos desbloquear por Enter
                self.get_logger().warn('No hay stdin disponible para Enter (EOF). Seguiré bloqueado.')
                return

            with self._lock:
                if self.state == self.WAIT_ENTER:
                    self.state = self.IDLE

            self.get_logger().info('Desbloqueado. Listo para recibir /num_vueltas.')

        self._enter_thread = threading.Thread(target=wait_enter, daemon=True)
        self._enter_thread.start()

    def imu_update(self, msg):
        q = msg.pose.pose.orientation
        self.yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

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