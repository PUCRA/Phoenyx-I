from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # This function generates and returns the complete LaunchDescription
    # for the system. It allows switching between simulation mode and
    # real robot mode using a launch argument.

    
    # -----------------------------------------
    # Launch argument to select simulation or real robot
    # -----------------------------------------
    sim_arg = DeclareLaunchArgument(
        'simulation',
        default_value='false',
        description='Run in simulation mode (true/false)'
    )

    # Retrieve the value of the "simulation" launch argument
    simulation = LaunchConfiguration('simulation')

    # -----------------------------------------
    # Paths to launch files used on the real robot
    # -----------------------------------------

    # Bringup launch file (robot base, motors, basic hardware)
    bringup_launch = os.path.join(
        get_package_share_directory('osr_bringup'),
        'launch',
        'osr_mod_launch.py'
    )
     # LiDAR driver launch file
    lidar_launch = os.path.join(
        get_package_share_directory('rplidar_ros'),
        'launch',
        'rplidar_a2m8_launch.py'
    )

    # -----------------------------------------
    # Navigation node (always launched)
    # -----------------------------------------
    # This node contains the maze navigation logic and runs in both
    # simulation and real robot modes.
    maze_navigation_node = Node(
        package='osr_control_challenge',
        executable='maze_navigation',
        name='maze_navigation',
        output='screen'
    )

    # ----------------------------------------- NO FUNCIONA
    # Safety / emergency node (real robot only)
    # -----------------------------------------
    # This node is used only when running on the real robot.
    # It is NOT launched in simulation mode.
    killer_node = Node(
        package='osr_control_challenge',
        executable='killer_node',   # Defined in setup.py entry_points
        name='killer_node',
        output='screen',
        condition=UnlessCondition(simulation)
    )

    # -----------------------------------------
    # Launch description definition
    # -----------------------------------------
    return LaunchDescription([
        # Declare the simulation argument
        sim_arg,

                # -------------------------------------
        # Real robot launch files (disabled in simulation)
        # -------------------------------------

        # Robot bringup
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bringup_launch),
            condition=UnlessCondition(simulation)
        ),
        # LiDAR driver
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(lidar_launch),
            condition=UnlessCondition(simulation)
        ),

        # -------------------------------------
        # Application nodes
        # -------------------------------------

        # Maze navigation node (always active)
        maze_navigation_node,

        # Emergency / killer node (real robot only)
        killer_node,
    ])
