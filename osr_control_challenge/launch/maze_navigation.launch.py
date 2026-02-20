import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # =========================
    # Launch arguments
    # =========================

    # Select which maze world to load in Gazebo (forwarded to maze_simulation.launch.py)
    maze_arg = DeclareLaunchArgument(
        'maze',
        default_value='maze_1.world',
        description='Gazebo world (.world) file to load (e.g., maze_1.world, maze_2.world)'
    )

    # Whether to launch RViz
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz'
    )

    maze = LaunchConfiguration('maze')
    rviz = LaunchConfiguration('rviz')
    simulation = True

    # =========================
    # Gazebo + robot bringup
    # =========================

    maze_simulation_launch = os.path.join(
        get_package_share_directory('osr_bringup'),
        'launch',
        'maze_simulation.launch.py'
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(maze_simulation_launch),
        launch_arguments={
            'maze': maze,
        }.items()
    )

    # =========================
    # Maze navigation node
    # =========================

    maze_navigation_node = Node(
        package='osr_control_challenge',
        executable='maze_navigation',
        name='maze_navigation',
        output='screen',
        parameters=[{
            'simulation': simulation
        }]
    )

    # =========================
    # RViz (optional)
    # =========================
    # Reuse the RViz config that you already know works.
    rviz_config = os.path.join(
        get_package_share_directory('osr_bringup'),
        'rviz',
        'nav_config.rviz'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=IfCondition(rviz),
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
    )

    # =========================
    # Launch description
    # =========================

    return LaunchDescription([
        maze_arg,
        rviz_arg,

        gazebo_launch,
        rviz_node,
        maze_navigation_node,
    ])
