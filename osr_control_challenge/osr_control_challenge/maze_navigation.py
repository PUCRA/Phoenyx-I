# =========================
# IMPORTS
# =========================

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist          # Message for linear 
from sensor_msgs.msg import LaserScan        # LIDAR message
import numpy as np
import math

# =========================
# FUNCTION TO ROTATE THE LASERSCAN
# =========================
# The LIDAR is physically mounted rotated (in our case 90°).
# This function repositions its measurements
# so that the 0° angle matches with the actual front of the robot.
def rotate_laserscan(scan_msg: LaserScan, angle_shift_rad: float) -> LaserScan:

    # scan_msg.ranges is a list:
    # each position corresponds to a specific angular direction
    ranges = np.array(scan_msg.ranges)

    # Angle between two consecutive LIDAR measurements (rad)
    angle_increment = scan_msg.angle_increment

    # We convert the rotation angle (in our case 90°)
    # into an index shift of the array
    #
    # Example:
    # - if each measurement is 1°
    # - and we rotate -90°
    # → we shift the array 90 positions
    shift = int(angle_shift_rad / angle_increment)

    # We move the distance array in a circular motion.
    # This causes:
    # - what was previously on one side
    # - to now be in front (angle 0°)
    rotated_ranges = np.roll(ranges, shift)

    # We create a new LaserScan message
    # with the distances already corrected
    corrected_scan = LaserScan()

    # We copy the original header
    corrected_scan.header = scan_msg.header

    # We indicate that the data is now aligned
    # with the robot's base_link frame.
    corrected_scan.header.frame_id = 'base_link'

    # The rest of the parameters remain unchanged.
    corrected_scan.angle_min = scan_msg.angle_min
    corrected_scan.angle_max = scan_msg.angle_max
    corrected_scan.angle_increment = scan_msg.angle_increment
    corrected_scan.time_increment = scan_msg.time_increment
    corrected_scan.scan_time = scan_msg.scan_time
    corrected_scan.range_min = scan_msg.range_min
    corrected_scan.range_max = scan_msg.range_max

    # We assign the already rotated distances
    corrected_scan.ranges = rotated_ranges.tolist()

    # The intensities do not need to be rotated.
    corrected_scan.intensities = scan_msg.intensities

    # We return the corrected scan
    return corrected_scan

# =========================
# MAIN MAZE NAVIGATION NODE
# =========================
class Maze_Navigation_Node(Node):
    def _init_(self):
        # Initializing the ROS 2 node
        super()._init_("maze_navigation_node")
        self.get_logger().info("🚀 Movement node initiated")

        # Robot speed Publisher
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # LIDAR Subscriber
        self.pose_subscriber = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        # =========================
        # ANGULAR PID CONTROL (FOLLOW CORRIDOR)
        # =========================
        # Error = left distance - right distance
        # error > 0 → robot closer to the right → turn left
        # error < 0 → robot closer to the left → turn right
        self.kp_ang = 1.0
        self.ki_ang = 0.0
        self.kd_ang = 0.0

        # Internal variables of the PID
        self.angular_integral = 0.0
        self.angular_prev_error = 0.0
        self.last_time = self.get_clock().now()

        # Angular velocity limit
        self.max_angular_speed = 0.6

    # =========================
    # LASER CALLBACK
    # =========================
    def scan_callback(self, msg: LaserScan):
        # We rotate the scan -90° to align the front of the robot.
        scan = rotate_laserscan(msg, math.radians(-90))

        # We convert ranges and angles to numpy
        ranges = np.array(scan.ranges)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))

        # Speed message
        cmd = Twist()

        # =========================
        # TIME CALCULATION FOR THE PID
        # =========================
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9

        # Avoid unusual dt values
        if dt <= 0.0 or dt > 1.0:
            dt = 0.1
        self.last_time = now


        # =========================
        # FILTERING INVALID DATA  (infinite values / NaN)
        # =========================
        mask_valid = np.isfinite(ranges)
        ranges = ranges[mask_valid]
        angles = angles[mask_valid]

        if len(ranges) == 0:
            self.get_logger().warn("🚧 There is no valid data.")
            return

        # =========================
        # SMOOTHING OF MEASUREMENTS
        # =========================
        smooth_ranges = np.convolve(ranges, np.ones(3) / 3, mode='same')

        new_ranges = []
        new_angles = []

        # We filter only useful distances
        for i in range(len(smooth_ranges)):
            if 0.05 < smooth_ranges[i] < 2.6:  # usable range
                new_ranges.append(smooth_ranges[i])
                new_angles.append(angles[i])

        if len(new_ranges) == 0:
            self.get_logger().warn("🚧 No valid points after filtering.")
            return

        new_ranges = np.array(new_ranges)
        new_angles = np.array(new_angles)

        # =========================
        # FRONTAL DISTANCE
        # =========================
        # Used to determine linear velocity
        mask_front = (new_angles >= np.radians(-5)) & (new_angles <= np.radians(5))
        front_candidates = new_ranges[mask_front]
        front_distance = np.nanmin(front_candidates) if front_candidates.size > 0 else 10.0

        # =========================
        # LATERAL DISTANCES
        # =========================
        # Right: 10° to 70°
        # Left: -10° to -70°
        mask_right = (new_angles >= np.radians(10)) & (new_angles <= np.radians(70))
        mask_left = (new_angles <= np.radians(-10)) & (new_angles >= np.radians(-70))

        right_vals = new_ranges[mask_right]
        left_vals = new_ranges[mask_left]

        # Default distance if no wall is detected
        nominal_dist = 1.0

        right_distance = np.mean(right_vals) if right_vals.size > 0 else nominal_dist
        left_distance = np.mean(left_vals) if left_vals.size > 0 else nominal_dist

        # Limit extreme values
        right_distance = float(np.clip(right_distance, 0.1, 2.5))
        left_distance = float(np.clip(left_distance, 0.1, 2.5))

        # =========================
        # ERROR IN FOLLOWING THE CENTER OF THE CORRIDOR
        # =========================
        # Compute the lateral error as the difference between left and right distances.
        # Positive error means the robot is closer to the right wall, negative means closer to the left.
        error = left_distance - right_distance

        # =========================
        # ANGULAR PID CONTROL
        # =========================
        # Integrate error over time for the integral term
        self.angular_integral += error * dt

        # Anti-windup: limit the integral term to prevent excessive overshoot
        integral_limit = 2.0
        self.angular_integral = max(-integral_limit,
                                    min(integral_limit, self.angular_integral))

        # Compute derivative term as the rate of change of error
        derivative = (error - self.angular_prev_error) / dt
        self.angular_prev_error = error

        # PID control output: proportional + integral + derivative
        u = (
            self.kp_ang * error
            + self.ki_ang * self.angular_integral
            + self.kd_ang * derivative
        )

        # Limit angular velocity to the robot's maximum capabilities
        cmd.angular.z = float(np.clip(u, -self.max_angular_speed,
                                      self.max_angular_speed))

        # =========================
        # LINEAR SPEED ACCORDING TO FRONTAL OBSTACLE
        # =========================
        # Adjust forward speed based on distance to obstacle in front
        # Stops if very close, slows down if approaching, moves normally otherwise
        if front_distance < 0.3:
            cmd.linear.x = 0.0
        elif front_distance < 0.8:
            cmd.linear.x = 0.10
        elif front_distance < 1.5:
            cmd.linear.x = 0.18
        else:
            cmd.linear.x = 0.25

        # Invert speed because the wheel motors respond inversely
        cmd.linear.x *= -1

        # We publish the speed command
        self.cmd_vel_publisher.publish(cmd)

        # =========================
        # DEPURATION LOG
        # =========================
        self.get_logger().info(
            f"🎯 front={front_distance:.2f}m "
            f"L={left_distance:.2f}m R={right_distance:.2f}m "
            f"err={error:.2f} "
            f"vel=({cmd.linear.x:.2f}, {cmd.angular.z:.2f})"
        )

# =========================
# MAIN FUNCTION
# =========================
def main(args=None):
    rclpy.init(args=args)
    node = Maze_Navigation_Node()
    rclpy.spin(node)
    rclpy.shutdown()