import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    # =========================
    # RPLIDAR launch (REAL) after 10s
    # =========================

    rplidar_launch_path = os.path.join(
        get_package_share_directory('rplidar_ros'),
        'launch',
        'rplidar_a2m8_launch.py'
    )

    rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rplidar_launch_path),
        launch_arguments={
            'serial_port': '/dev/ttyUSB0',
            'serial_baudrate': '115200',
            'scan_mode': 'Standard'
        }.items()
    )

    # =========================
    # Maze navigation node (REAL)
    # =========================

    maze_navigation_node = Node(
        package='osr_control_challenge',
        executable='maze_navigation_real',
        name='maze_navigation_real',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    # =========================
    # Launch description
    # =========================

    return LaunchDescription([
        LogInfo(msg='[osr_control_real] Starting RPLIDAR...'),
        rplidar_launch,

        LogInfo(msg='[osr_control_real] Starting maze_navigation (REAL)...'),
        maze_navigation_node,
    ])
