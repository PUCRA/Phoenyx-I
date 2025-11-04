import math
from functools import partial

import rclpy
from rclpy.parameter import Parameter
from rclpy.node import Node
import tf2_ros

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, TwistWithCovariance, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from osr_interfaces.msg import CommandDrive, CommandCorner

import json
import collections
from collections import deque
from statistics import median


class Rover(Node):
    """Math and motor control algorithms to move the rover"""

    def __init__(self):
        super().__init__("rover")
        self.log = self.get_logger()
        self.log.info("Initializing Rover")

        self.heading_buffer = deque(maxlen=5)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('rover_dimensions.d1', Parameter.Type.DOUBLE),
                ('rover_dimensions.d2', Parameter.Type.DOUBLE),
                ('rover_dimensions.d3', Parameter.Type.DOUBLE),
                ('rover_dimensions.d4', Parameter.Type.DOUBLE),
                ('rover_dimensions.wheel_radius', Parameter.Type.DOUBLE),
                ('drive_no_load_rpm', Parameter.Type.DOUBLE),
                ('enable_odometry', Parameter.Type.BOOL)
            ]
        )
        self.d1 = self.get_parameter('rover_dimensions.d1').get_parameter_value().double_value
        self.d2 = self.get_parameter('rover_dimensions.d2').get_parameter_value().double_value
        self.d3 = self.get_parameter('rover_dimensions.d3').get_parameter_value().double_value
        self.d4 = self.get_parameter('rover_dimensions.d4').get_parameter_value().double_value

        self.min_radius = 0.45  # [m]
        self.max_radius = 6.4  # [m]

        self.no_cmd_thresh = 0.05  # [rad]
        self.wheel_radius = self.get_parameter(
            "rover_dimensions.wheel_radius").get_parameter_value().double_value  # [m]
        drive_no_load_rpm = self.get_parameter(
            "drive_no_load_rpm").get_parameter_value().double_value
        self.max_vel = self.wheel_radius * drive_no_load_rpm / 60 * 2 * math.pi  # [m/s]
        self.should_calculate_odom = self.get_parameter("enable_odometry").get_parameter_value().bool_value

        if self.should_calculate_odom:
            self.get_logger().info("Calculating wheel odometry and publishing to /odom topic")
            self.odometry = Odometry()
            self.odometry.header.stamp = self.get_clock().now().to_msg()
            self.odometry.header.frame_id = "odom"
            self.odometry.child_frame_id = "base_link"
            self.new_angle = 0.0  # Inicializa el ángulo en cero

        self.curr_positions = {}
        self.curr_velocities = {}
        self.curr_twist = TwistWithCovariance()
        self.curr_turning_radius = self.max_radius

        self.cmd_vel_sub = self.create_subscription(Twist, "/cmd_vel",
                                                    partial(self.cmd_cb, intuitive=False), 1)
        self.cmd_vel_int_sub = self.create_subscription(Twist, "/cmd_vel_intuitive",
                                                        partial(self.cmd_cb, intuitive=True), 1)
        self.drive_enc_sub = self.create_subscription(JointState, "/drive_state", self.enc_cb, 1)
        self.corner_enc_sub = self.create_subscription(JointState, "/corner_state", self.enc_cb, 1)

        self.turning_radius_pub = self.create_publisher(Float64, "/turning_radius", 1)
        if self.should_calculate_odom:
            self.odometry_pub = self.create_publisher(Odometry, "/odom", 2)
            self.aruco_sub = self.create_subscription(Twist, "/aruco_pos", self.aruco_callback, 1)
            self.tf_pub = tf2_ros.TransformBroadcaster(self)
            self.angle_offset = 0.0

        self.corner_cmd_pub = self.create_publisher(CommandCorner, "/cmd_corner", 1)
        self.drive_cmd_pub = self.create_publisher(CommandDrive, "/cmd_drive", 1)

        self.get_logger().info("Rover initialized")
        self.buffer = collections.deque(maxlen=10)

    def cmd_cb(self, twist_msg, intuitive=False):
        MAX_ANGULAR_SPEED = 1.4
        MAX_LINEAR_SPEED = 0.6
        MIN_LINEAR_SPEED = 0.2
        MIN_ANGULAR_SPEED = 0.8
        MIN_SPEED = 0.3

        v = twist_msg.linear.x
        w = twist_msg.angular.z
        magnitude = math.sqrt(v**2 + w**2)

        if magnitude < 1e-3:
            v = 0.0
            w = 0.0
        elif (abs(v) < 0.08 and abs(w) > 0.3) or (abs(v) == 0.0 and abs(w) > 0.0):
            v = 0.0
            if abs(w) < MIN_ANGULAR_SPEED and w != 0.0:
                w = abs(w)/w*MIN_ANGULAR_SPEED
            w = max(min(w, MAX_ANGULAR_SPEED), -MAX_ANGULAR_SPEED)
        elif magnitude < MIN_SPEED and magnitude != 0.0:
            norm_v = v / magnitude
            norm_w = w / magnitude
            scaled_v = norm_v * MIN_LINEAR_SPEED
            scaled_w = norm_w * MIN_ANGULAR_SPEED
        scaled_v = max(min(v, MAX_LINEAR_SPEED), -MAX_LINEAR_SPEED)
        scaled_w = max(min(w, MAX_ANGULAR_SPEED), -MAX_ANGULAR_SPEED)

        twist_msg.angular.z = scaled_w
        twist_msg.angular.x = scaled_v

        if twist_msg.angular.x == 0.0 and abs(twist_msg.angular.z) > 0.05:
            corner_cmd_msg, drive_cmd_msg = self.calculate_rotate_in_place_cmd(twist_msg)
        else:
            desired_turning_radius = self.twist_to_turning_radius(twist_msg, intuitive_mode=intuitive)
            corner_cmd_msg = self.calculate_corner_positions(desired_turning_radius)
            max_vel = abs(desired_turning_radius) / (abs(desired_turning_radius) + self.d1) * self.max_vel
            if math.isnan(max_vel):
                max_vel = self.max_vel
            velocity = min(max_vel, twist_msg.linear.x)
            drive_cmd_msg = self.calculate_drive_velocities(velocity, desired_turning_radius)

        self.corner_cmd_pub.publish(corner_cmd_msg)
        self.drive_cmd_pub.publish(drive_cmd_msg)

    def enc_cb(self, msg):
        self.curr_positions = {**self.curr_positions, **dict(zip(msg.name, msg.position))}
        self.curr_velocities = {**self.curr_velocities, **dict(zip(msg.name, msg.velocity))}

        if self.should_calculate_odom and len(self.curr_positions) == 10:
            now = self.get_clock().now()
            dt = float(now.nanoseconds - (self.odometry.header.stamp.sec*10**9 + self.odometry.header.stamp.nanosec)) / 10**9
            self.forward_kinematics()
            dx = self.curr_twist.twist.linear.x * dt
            dth = self.curr_twist.twist.angular.z * dt

            self.new_angle += dth
            self.new_angle += self.angle_offset
            self.odometry.pose.pose.orientation.z = math.sin(self.new_angle / 2.)
            self.odometry.pose.pose.orientation.w = math.cos(self.new_angle / 2.)

            self.odometry.pose.pose.position.x += math.cos(self.new_angle) * dx
            self.odometry.pose.pose.position.y += math.sin(self.new_angle) * dx

            self.odometry.pose.covariance = 36 * [0.0,]
            self.odometry.twist = self.curr_twist
            self.odometry.header.stamp = now.to_msg()

            self.odometry_pub.publish(self.odometry)
            transform_msg = TransformStamped()
            transform_msg.header.frame_id = "odom"
            transform_msg.child_frame_id = "base_link"
            transform_msg.header.stamp = now.to_msg()
            transform_msg.transform.translation.x = self.odometry.pose.pose.position.x
            transform_msg.transform.translation.y = self.odometry.pose.pose.position.y
            transform_msg.transform.rotation = self.odometry.pose.pose.orientation
            self.tf_pub.sendTransform(transform_msg)

    def calculate_drive_velocities(self, speed, current_radius):
        speed = max(-self.max_vel, min(self.max_vel, speed))
        cmd_msg = CommandDrive()
        if speed == 0:
            return cmd_msg
        elif abs(current_radius) >= self.max_radius:
            angular_vel = speed / self.wheel_radius
            cmd_msg.left_front_vel = angular_vel
            cmd_msg.left_middle_vel = angular_vel
            cmd_msg.left_back_vel = angular_vel
            cmd_msg.right_back_vel = -angular_vel
            cmd_msg.right_middle_vel = -angular_vel
            cmd_msg.right_front_vel = -angular_vel
            return cmd_msg
        else:
            radius = abs(current_radius)
            angular_velocity_center = float(speed) / radius
            vel_middle_closest = (radius - self.d4) * angular_velocity_center
            vel_corner_closest = math.hypot(radius - self.d1, self.d3) * angular_velocity_center
            vel_corner_farthest = math.hypot(radius + self.d1, self.d3) * angular_velocity_center
            vel_middle_farthest = (radius + self.d4) * angular_velocity_center

            ang_vel_middle_closest = vel_middle_closest / self.wheel_radius
            ang_vel_corner_closest = vel_corner_closest / self.wheel_radius
            ang_vel_corner_farthest = vel_corner_farthest / self.wheel_radius
            ang_vel_middle_farthest = vel_middle_farthest / self.wheel_radius

            if current_radius > 0:
                cmd_msg.left_front_vel = ang_vel_corner_closest
                cmd_msg.left_back_vel = ang_vel_corner_closest
                cmd_msg.left_middle_vel = ang_vel_middle_closest
                cmd_msg.right_back_vel = -ang_vel_corner_farthest
                cmd_msg.right_front_vel = -ang_vel_corner_farthest
                cmd_msg.right_middle_vel = -ang_vel_middle_farthest
            else:
                cmd_msg.left_front_vel = ang_vel_corner_farthest
                cmd_msg.left_back_vel = ang_vel_corner_farthest
                cmd_msg.left_middle_vel = ang_vel_middle_farthest
                cmd_msg.right_back_vel = -ang_vel_corner_closest
                cmd_msg.right_front_vel = -ang_vel_corner_closest
                cmd_msg.right_middle_vel = -ang_vel_middle_closest

            return cmd_msg

    def calculate_corner_positions(self, radius):
        cmd_msg = CommandCorner()
        if radius >= self.max_radius:
            return cmd_msg

        theta_front_closest = math.atan2(self.d3, abs(radius) - self.d1)
        theta_front_farthest = math.atan2(self.d3, abs(radius) + self.d1)

        if radius > 0:
            cmd_msg.left_front_pos = -theta_front_closest
            cmd_msg.left_back_pos = theta_front_closest
            cmd_msg.right_back_pos = theta_front_farthest
            cmd_msg.right_front_pos = -theta_front_farthest
        else:
            cmd_msg.left_front_pos = theta_front_farthest
            cmd_msg.left_back_pos = -theta_front_farthest
            cmd_msg.right_back_pos = -theta_front_closest
            cmd_msg.right_front_pos = theta_front_closest

        return cmd_msg

    def calculate_rotate_in_place_cmd(self, twist):
        corner_cmd = CommandCorner()
        corner_cmd.left_front_pos = math.atan(self.d3/self.d1)
        corner_cmd.left_back_pos = -corner_cmd.left_front_pos
        corner_cmd.right_back_pos = math.atan(self.d2/self.d1)
        corner_cmd.right_front_pos = -corner_cmd.right_back_pos

        drive_cmd = CommandDrive()
        if abs(twist.angular.z) < 0.6:
            twist.angular.z = abs(twist.angular.z)/twist.angular.z*0.6
        angular_vel = -twist.angular.z
        front_wheel_vel = math.hypot(self.d1, self.d3) * angular_vel / self.wheel_radius
        drive_cmd.left_front_vel = front_wheel_vel
        drive_cmd.right_front_vel = front_wheel_vel
        back_wheel_vel = math.hypot(self.d1, self.d2) * angular_vel / self.wheel_radius
        drive_cmd.left_back_vel = back_wheel_vel
        drive_cmd.right_back_vel = back_wheel_vel
        middle_wheel_vel = self.d4 * angular_vel / self.wheel_radius
        drive_cmd.left_middle_vel = middle_wheel_vel
        drive_cmd.right_middle_vel = middle_wheel_vel

        return corner_cmd, drive_cmd

    def twist_to_turning_radius(self, twist, clip=True, intuitive_mode=False):
        try:
            if intuitive_mode and twist.linear.x < 0:
                radius = twist.linear.x / -twist.angular.z
            else:
                radius = twist.linear.x / twist.angular.z
        except ZeroDivisionError:
            return float("Inf")

        if not clip:
            return radius
        if radius == 0:
            if intuitive_mode:
                if twist.angular.z == 0:
                    return self.max_radius
                else:
                    radius = self.min_radius * self.max_vel / twist.angular.z
            else:
                return self.max_radius
        if radius > 0:
            radius = max(self.min_radius, min(self.max_radius, radius))
        else:
            radius = max(-self.max_radius, min(-self.min_radius, radius))

        return radius

    def angle_to_turning_radius(self, angle):
        try:
            radius = self.d3 / math.tan(angle)
        except ZeroDivisionError:
            return float("Inf")
        return radius

    def forward_kinematics(self):
        theta_fl = -self.curr_positions['corner_left_front']
        theta_fr = -self.curr_positions['corner_right_front']
        theta_bl = -self.curr_positions['corner_left_back']
        theta_br = -self.curr_positions['corner_right_back']

        if theta_fl + theta_fr + theta_bl + theta_br > 0:
            r_front_closest = self.d1 + self.angle_to_turning_radius(theta_fl)
            r_front_farthest = -self.d1 + self.angle_to_turning_radius(theta_fr)
            r_back_closest = -self.d1 - self.angle_to_turning_radius(theta_bl)
            r_back_farthest = self.d1 - self.angle_to_turning_radius(theta_br)
        else:
            r_front_farthest = self.d1 + self.angle_to_turning_radius(theta_fl)
            r_front_closest = -self.d1 + self.angle_to_turning_radius(theta_fr)
            r_back_farthest = -self.d1 - self.angle_to_turning_radius(theta_bl)
            r_back_closest = self.d1 - self.angle_to_turning_radius(theta_br)

        approx_turning_radius = sum(sorted([r_front_farthest, r_front_closest, r_back_farthest, r_back_closest])[1:3])/2.0
        if math.isnan(approx_turning_radius):
            approx_turning_radius = self.max_radius
        self.curr_turning_radius = approx_turning_radius

        drive_angular_velocity = (self.curr_velocities['drive_left_middle'] + self.curr_velocities['drive_right_middle']) / 2.
        self.curr_twist.twist.linear.x = drive_angular_velocity * self.wheel_radius
        try:
            self.curr_twist.twist.angular.z = self.curr_twist.twist.linear.x / self.curr_turning_radius
        except ZeroDivisionError:
            self.curr_twist.twist.angular.z = 0.

    def aruco_callback(self, msg):
        self.odometry.pose.pose.position.x = msg.linear.x
        self.odometry.pose.pose.position.y = msg.linear.y
        self.angle_offset = msg.angular.z - self.new_angle


def main(args=None):
    rclpy.init(args=args)
    rover = Rover()
    rclpy.spin(rover)
    rover.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
