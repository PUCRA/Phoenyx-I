from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    """
    Launch file for real robot maze navigation.
    Launches: hardware bringup, LiDAR, and maze navigation node.
    """
    
    # -----------------------------------------
    # Paths for real robot launch files
    # -----------------------------------------
    real_bringup_launch = os.path.join(
        get_package_share_directory('osr_bringup'),
        'launch',
        'osr_mod_launch.py'
    )
    
    real_lidar_launch = os.path.join(
        get_package_share_directory('rplidar_ros'),
        'launch',
        'rplidar_a2m8_launch.py'
    )
    
    # -----------------------------------------
    # Include launch files
    # -----------------------------------------
    real_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(real_bringup_launch)
    )
    
    real_lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(real_lidar_launch)
    )
    
    # -----------------------------------------
    # Maze navigation node
    # -----------------------------------------
    maze_navigation_node = Node(
        package='osr_control_challenge',
        executable='maze_navigation',
        name='maze_navigator',
        output='screen',
        parameters=[{
            'simulation': False  # Robot real
        }]
    )
    
    # -----------------------------------------
    # Launch description
    # -----------------------------------------
    return LaunchDescription([
        real_bringup,
        real_lidar,
        maze_navigation_node,
    ])