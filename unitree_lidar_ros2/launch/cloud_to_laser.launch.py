import os
import subprocess
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    package_name = 'unitree_lidar_ros2'
    
    # Intentar cargar archivo de configuración del converter
    try:
        package_share = get_package_share_directory(package_name)
        config_file = os.path.join(package_share, 'config', 'pointcloud_to_laserscan_params.yaml')
        use_config = os.path.exists(config_file)
    except:
        use_config = False
        config_file = None

    # ================================================================
    # NODO DEL LIDAR
    # ================================================================
    lidar_node = Node(
        package='unitree_lidar_ros2',
        executable='unitree_lidar_ros2_node',
        name='unitree_lidar_ros2_node',
        output='screen',
        parameters=[
            {'initialize_type': 2},
            {'work_mode': 0},
            {'use_system_timestamp': True},
            {'range_min': 0.0},
            {'range_max': 100.0},
            {'cloud_scan_num': 18},
            {'serial_port': '/dev/ttyACM0'},
            {'baudrate': 2000000},
            {'lidar_port': 6101},
            {'lidar_ip': '192.168.1.62'},
            {'local_port': 6201},
            {'local_ip': '192.168.1.100'},
            {'cloud_frame': "unilidar_lidar"},
            {'cloud_topic': "unilidar/cloud"},
            {'imu_frame': "unilidar_imu"},
            {'imu_topic': "unilidar/imu"},
        ]
    )

    # ================================================================
    # NODO CONVERTER (PointCloud2 → LaserScan)
    # ================================================================
    # Configuración MINIMALISTA para trabajar SOLO con el sensor
    converter_params = {
        'min_height': 0.0,        # Medio metro por debajo
        'max_height': 0.25,         # Medio metro por encima
        'angle_min': -3.141592653589793,  # -180°
        'angle_max': 3.141592653589793,   # +180°
        'angle_increment': 0.004363323,   # 0.25° (buena resolución)
        'range_min': 0.05,         # 5cm mínimo
        'range_max': 10.0,         # 10m máximo
        'rotation_offset': 1.5707963,    # +90°
        'target_frame': 'unilidar_lidar',  # Frame del LIDAR (NO base_link)
    }

    if use_config:
        converter_node = Node(
            package='unitree_lidar_ros2',
            executable='pointcloud_to_laserscan',
            name='pointcloud_to_laserscan',
            output='screen',
            parameters=[config_file]
        )
    else:
        converter_node = Node(
            package='unitree_lidar_ros2',
            executable='pointcloud_to_laserscan',
            name='pointcloud_to_laserscan',
            output='screen',
            parameters=[converter_params]
        )

    # ================================================================
    # RVIZ (OPCIONAL)
    # ================================================================
    try:
        package_path = subprocess.check_output(
            ['ros2', 'pkg', 'prefix', 'unitree_lidar_ros2']
        ).decode('utf-8').rstrip()
        
        rviz_config_file = os.path.join(
            package_path, 'share', 'unitree_lidar_ros2', 'view.rviz'
        )
        
        print("rviz_config_file = " + rviz_config_file)
        
        rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='log'
        )
        
        return LaunchDescription([
            lidar_node,
            converter_node,
            rviz_node
        ])
        
    except:
        # Si RViz no está disponible, lanzar solo LIDAR y converter
        return LaunchDescription([
            lidar_node,
            converter_node
        ])