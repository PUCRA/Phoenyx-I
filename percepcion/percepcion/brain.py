import os
import time
import traceback
from collections import Counter
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import pyrealsense2 as rs

# Custom module for processing
from percepcion.Recorte2number import Recorte2number


class BrainIntelNode(Node):
    def __init__(self):
        # Initialize the ROS 2 node with the given name
        super().__init__('brain_intel_node')

        # ------------------ Parameters ------------------
        # Declare configurable ROS parameters
        self.declare_parameters(namespace='', parameters=[
            ('numero_muestras', 10),   # Number of samples to collect before deciding
            ('show_camera', False),    # If you want another window with the camera view
        ])

        # Read parameter values
        self.numero_muestras = int(self.get_parameter('numero_muestras').value)
        self.show_camera = bool(self.get_parameter('show_camera').value)

        # Log the number of samples being used
        self.get_logger().info(f"Número de muestras: {self.numero_muestras}")

        # ------------------ Publishers ------------------
        # Publisher that outputs the final number of rotations
        self.pub_vueltas = self.create_publisher(Int32, '/num_vueltas', 10)
        # Publisher that streams the RGB image as a ROS topic
        self.intel_pub = self.create_publisher(Image, 'rgb_frame', 10)
        # OpenCV <-> ROS image bridge
        self.br_rgb = CvBridge()

        # ------------------ Processors ------------------
        # Object responsible for detecting number and color from images
        self.converter = Recorte2number()

        # ------------------ Internal variables ------------------
        # List of detected numbers across samples
        self.numeros = []
        # List of detected colors across samples
        self.colores = []
        # Counter of how many samples have been processed
        self.conteo_muestras = 0
        # Finite State Machine (FSM) state
        self.estado = 1
        # Final decided number
        self.numero_final = 0
        # Final decided color
        self.color_final = ""
        # Timestamp when the sampling phase started
        self.ini_time = time.time()
        # Reference number used for debug image naming
        self.numero_really = 5
        # Index for saved debug images
        self.i = 0
        # Deadline timestamp used in reset state
        self._reset_deadline = None

        # ------------------ Initialize RealSense ------------------
        try:
            # Create RealSense pipeline and configuration
            self.pipe = rs.pipeline()
            self.cfg = rs.config()
            # Enable RGB color stream
            self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            # Start the camera
            self.pipe.start(self.cfg)
            self.camera_ready = True
            self.get_logger().info("RealSense inicializada.")
        except Exception as e:
            # Handle camera connection errors
            self.get_logger().error(f"RealSense NO conectada: {e}")
            traceback.print_exc()
            self.camera_ready = False
            self.pipe = None

        # ------------------ Visualization window ------------------
        # Create an OpenCV window to show the camera feed
        if self.show_camera:
            cv2.namedWindow("RealSense View", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("RealSense View", 640, 480)

        # ------------------ Capture thread ------------------
        # Flag to control the camera loop execution
        self._running = True
        # Separate thread for camera capture to avoid blocking ROS callbacks
        self.capture_thread = threading.Thread(
            target=self.camera_loop,
            daemon=True
        )
        self.capture_thread.start()

        # ------------------ FSM timer ------------------
        # Timer that periodically executes the FSM logic
        self.fsm_timer = self.create_timer(0.001, self.FSM)

    # ============================================================================================
    #                                   REALSENSE LOOP
    # ============================================================================================
    def camera_loop(self):
        # Exit if the camera is not properly initialized
        if not self.camera_ready:
            self.get_logger().error("Cámara no lista")
            return

        # Main camera acquisition loop
        while self._running:
            try:
                # Wait for a new set of frames from the camera
                frames = self.pipe.wait_for_frames(timeout_ms=1000)
                # Extract the color frame
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue

                # Convert frame data to a NumPy array
                color_image = np.asanyarray(color_frame.get_data())

                # Skip invalid images
                if color_image is None or color_image.size == 0:
                    continue

                # Publish the image as a ROS message
                ros_img = self.br_rgb.cv2_to_imgmsg(color_image, encoding='bgr8')
                self.intel_pub.publish(ros_img)

                # Process the image only if sampling is enabled by the FSM
                if self.enable_muestras:
                    self.tratar_recorte(color_image)

                # Display the camera feed
                if self.show_camera:
                    cv2.imshow("RealSense View", color_image)
                    cv2.waitKey(1)

            except Exception as e:
                # Log and print any runtime errors in the camera loop
                self.get_logger().error(f"Error en camera_loop: {e}")
                traceback.print_exc()

        # Log when the camera loop stops
        self.get_logger().info("camera_loop detenido.")

    # ============================================================================================
    #                                   PROCESSING
    # ============================================================================================
    def tratar_recorte(self, image):
        
        # Ignore invalid images
        if image is None or image.size == 0:
            return

        try:
            # Extract detected number, color, and thresholded image
            res = self.converter.obtener_colorYnum(image)
        except Exception as e:
            # Warn if the detection process fails
            self.get_logger().warning(f"Error obtener_colorYnum: {e}")
            return

        # Validate the returned result
        if not res or not isinstance(res, tuple):
            return

        # Unpack detection results
        numero, color, img_thresh = res

        # If a number was detected, store it
        if numero is not None:
            self.numeros.append(numero)
            self.get_logger().info(f"Detectado número {numero}, color {color}")
            self.i += 1

        # If a color was detected, store it
        if color:
            self.colores.append(color)

        # Increase the sample counter
        self.conteo_muestras += 1

    # ============================================================================================
    #                                   DECISION
    # ============================================================================================
    def decision_making(self):
        # If no numbers were detected, return a default result
        if not self.numeros:
            return 0, "Distractorio"

        # Select the most common detected number
        numero = Counter(self.numeros).most_common(1)[0][0]

        # Count occurrences of each color
        counts = {
            "Rojo": self.colores.count("Rojo"),
            "Azul": self.colores.count("Azul"),
            "Distractorio": len(self.colores)
        }
        # Select the color with the highest count
        color = max(counts, key=counts.get)

        return numero, color

    # ============================================================================================
    #                                   FSM
    # ============================================================================================
    def FSM(self):
        try:
            if self.estado == 1:  # COLLECTION STATE
                # Enable sample collection
                self.enable_muestras = True
                # Transition if enough samples or timeout reached
                if self.conteo_muestras >= self.numero_muestras or \
                   (time.time() - self.ini_time) >= 40:
                    self.estado = 2

            elif self.estado == 2:  # PROCESSING STATE
                # Stop collecting samples
                self.enable_muestras = False
                # Make the final decision
                self.numero_final, self.color_final = self.decision_making()

                # Prepare the output message
                msg = Int32()
                if self.color_final == "Azul":
                    msg.data = int(self.numero_final)
                elif self.color_final == "Rojo":
                    msg.data = -int(self.numero_final)
                else:
                    msg.data = 0

                # Publish the result
                self.pub_vueltas.publish(msg)
                # Set a short delay before resetting
                self._reset_deadline = time.time() + 0.2
                self.estado = 3

            elif self.estado == 3:  # RESET STATE
                # Reset all variables after the deadline
                if time.time() >= self._reset_deadline:
                    self.numeros.clear()
                    self.colores.clear()
                    self.conteo_muestras = 0
                    self.ini_time = time.time()
                    self.estado = 1

        except Exception as e:
            # Log FSM-related errors
            self.get_logger().error(f"Error en FSM: {e}")
            traceback.print_exc()

    # ============================================================================================
    #                                   DESTRUCTOR
    # ============================================================================================
    def destroy_node(self):
        # Log node shutdown
        self.get_logger().info("Cerrando nodo...")
        # Stop the camera loop
        self._running = False
        time.sleep(0.1)

        # Stop the RealSense pipeline if active
        try:
            if self.pipe:
                self.pipe.stop()
        except:
            pass

        # Close all OpenCV windows
        try:
            cv2.destroyAllWindows()
        except:
            pass

        # Call the parent class destructor
        super().destroy_node()


# ============================================================================================
#                                       MAIN
# ============================================================================================
def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)
    # Create the node instance
    node = BrainIntelNode()
    try:
        # Keep the node running
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown
        node.destroy_node()
        rclpy.shutdown()


# Entry point when running the script directly
if __name__ == "__main__":
    main()
