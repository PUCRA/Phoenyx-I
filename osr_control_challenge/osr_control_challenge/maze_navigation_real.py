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
# MAIN MAZE NAVIGATION NODE
# =========================
class Maze_Navigation_Node(Node):
    def __init__(self):
        # Initializing the ROS 2 node
        super().__init__("maze_navigation")
        self.get_logger().info("🚀 Movement node initiated")

        # Robot speed Publisher
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # LIDAR Subscriber
        self.pose_subscriber = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        # =========================
        # ANGULAR PID CONTROL 
        # =========================
        self.kp_ang = 0.7
        self.ki_ang = 0.0
        self.kd_ang = 0.15

        # Internal variables of the PID
        self.angular_integral = 0.0
        self.angular_prev_error = 0.0
        self.last_time = self.get_clock().now()

        # Angular velocity limit
        self.max_angular_speed = 1.4

    # =========================
    # LASER CALLBACK
    # =========================
    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=np.inf)  # NaN → inf
        num_ranges = len(ranges)

        cmd = Twist()

        # =========================
        # TIME CALCULATION FOR PID
        # =========================
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0.0 or dt > 1.0:
            dt = 0.1
        self.last_time = now

        # =========================
        # FUNCTION: get min distance in angular window
        # =========================
        def get_sector_distance(center_deg, width_deg):
            center_rad = math.radians(center_deg)

            # Central index for the desired angle
            index_center = int(round((center_rad - msg.angle_min) / msg.angle_increment)) % num_ranges

            # Number of indices to cover the desired angular width
            half_window = int(round(math.radians(width_deg / 2) / msg.angle_increment))

            # List of indices for the sector
            indices = [(index_center + i) % num_ranges for i in range(-half_window, half_window + 1)]
            
            # Distance values for these indices
            sector_ranges = ranges[indices]

            # Minimum distance in the sector, ignoring invalid readings
            min_distance = float(np.min(sector_ranges))

            # Clip the distance to a reasonable range for safety
            min_distance = np.clip(min_distance, 0.1, 10.0)

            return min_distance

        # =========================
        # DISTANCES USING CORRECT INDEX METHOD
        # =========================
        front_distance = get_sector_distance(90, 20)  # 20° window around front
        left_distance  = get_sector_distance(150, 60) # 60° window around left
        right_distance = get_sector_distance(30, 60)  # 60° window around right

        # =========================
        # Dinamic adjustment of kp based on frontal distance
        # =========================
        # Additionally, the proportional gain (Kp) is adjusted dynamically
        # depending on the frontal distance. When the robot approaches to a frontal wall,
        # Kp is increased to make the angular response more aggressive and reactive.
        # When the path ahead is clear, Kp returns to its nominal value to ensure
        # smoother motion and reduce oscillations.

        safe_distance = 1.5   
        limit_distance  = 0.75
        
        min_kp = self.kp_ang
        max_kp = 2.0  

        if front_distance <= limit_distance:
            kp_dynamic = max_kp
        elif front_distance >= safe_distance:
            kp_dynamic = min_kp
        else:
            scale = (safe_distance - front_distance) / (safe_distance - limit_distance)
            kp_dynamic = min_kp + scale * (max_kp - min_kp)

        # =========================
        # ANGULAR PID CONTROL
        # =========================
        # The lateral error is computed as the difference between the distances
        # to the left and right walls. The objective is to keep this value near zero,
        # which corresponds to the robot being centered in the corridor.
        #
        # Interpretation:
        #   error > 0  → robot is closer to the right wall (needs correction to the left)
        #   error < 0  → robot is closer to the left wall (needs correction to the right)
        error = left_distance - right_distance
        
        # Integrate error over time for the integral term
        self.angular_integral += error * dt

        # Anti-windup: limit the integral term to prevent excessive overshoot
        integral_limit = 2.0
        self.angular_integral = max(-integral_limit, min(integral_limit, self.angular_integral))

        # Compute derivative term as the rate of change of error
        derivative = (error - self.angular_prev_error) / dt
        self.angular_prev_error = error

        # PID control output: proportional + integral + derivative
        u = (
            kp_dynamic * error
            + self.ki_ang * self.angular_integral
            + self.kd_ang * derivative
        )

        # Limit angular velocity to the robot's maximum capabilities
        cmd.angular.z = float(np.clip(u, -self.max_angular_speed, self.max_angular_speed))
        cmd.angular.z *= -1

        # =========================
        # LINEAR SPEED ACCORDING TO FRONTAL OBSTACLE
        # =========================
        # Adjust forward speed based on distance to obstacle in front
        # Stops if very close, slows down if approaching, moves normally otherwise
        stop_distance = 0.6 
        min_distance = stop_distance 
        max_distance = 3.0  
        
        min_speed = 0.2      
        max_speed = 0.5      

        d = front_distance

        if d <= stop_distance:
            cmd.linear.x = 0.0
        elif d >= max_distance:
            cmd.linear.x = max_speed
        else:
            # Escalado lineal entre min_distance y max_distance
            cmd.linear.x = min_speed + (d - min_distance) * (max_speed - min_speed) / (max_distance - min_distance)

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
    def stop_robot(self):
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        self.cmd_vel_publisher.publish(stop_cmd)
        self.get_logger().info("🚀 Shutdown complete")

# =========================
# MAIN FUNCTION
# =========================
def main(args=None):
    rclpy.init(args=args)
    node = Maze_Navigation_Node()
    rclpy.spin(node)
    rclpy.shutdown()