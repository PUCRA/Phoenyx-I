#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
import numpy as np

class LidarQuadrants(Node):
    def __init__(self):
        super().__init__('lidar_quadrants')

        # Suscripción al topic del LIDAR
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        # Ángulos centrales de los cuadrantes en grados
        self.quadrants_deg = {
            'front': 90,
            'left': 180,
            'back': 270,
            'right': 0
        }
        self.quadrant_width_deg = 30  # cada cuadrante ±22.5°

        self.get_logger().info("Nodo LidarQuadrants inicializado")

    def laser_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=np.inf)  # NaN → inf
        num_ranges = len(ranges)

        results = {}
        for name, center_angle_deg in self.quadrants_deg.items():
            # Convertimos el ángulo central a radianes
            center_angle_rad = math.radians(center_angle_deg)

            # Calculamos índice central
            index_center = int(round((center_angle_rad - msg.angle_min) / msg.angle_increment)) % num_ranges

            # Calculamos ancho del cuadrante en índices
            half_window = int(round(math.radians(self.quadrant_width_deg / 2) / msg.angle_increment))

            # Lista de índices para este cuadrante
            indices = [(index_center + i) % num_ranges for i in range(-half_window, half_window + 1)]

            # Distancia mínima dentro del cuadrante
            distances = ranges[indices]
            results[name] = float(np.min(distances))

        self.get_logger().info(
            f"Distances (m) | Front: {results['front']:.2f} | Left: {results['left']:.2f} | "
            f"Back: {results['back']:.2f} | Right: {results['right']:.2f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = LidarQuadrants()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
