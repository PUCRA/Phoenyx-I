import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

import xacro


def generate_launch_description():
    
    # Argument to choose which world file (maze) to load
    world_selector_arg = DeclareLaunchArgument(
        'maze',
        default_value='maze_1.world',
        description='Nombre del archivo .world a cargar en Gazebo'
    )

    # Build the path to the selected world file
    world_path = PathJoinSubstitution([
        FindPackageShare('osr_gazebo'),
        'worlds',
        LaunchConfiguration('maze')
    ])

    # Include the default Gazebo launch file with the selected world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={
            'world': world_path,
            'use_sim_time': 'true'
        }.items(),
    )

    # Load the robot description (URDF generated from a xacro file)
    osr_urdf_path = os.path.join(get_package_share_directory('osr_gazebo'))
    xacro_file = os.path.join(osr_urdf_path, 'urdf', 'osr_simplified.urdf.xacro')
    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc)
    params = {
        'robot_description': doc.toxml()
        ,'use_sim_time': True
        }

    # Nodes
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    # Controller loading commands
    controller_spawn = Node(
        package='osr_gazebo',
        executable='osr_controller',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'rover'],
        output='screen'
    )

    # Controladores
    # joint_state_controller
    load_joint_state_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster'],
        output='screen'
    )

    # wheel_velocity_controller
    rover_wheel_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'wheel_controller'],
        output='screen'
    )

    # servo_controller
    servo_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'servo_controller'],
        output='screen'
    )

    return LaunchDescription([
        world_selector_arg, 
        controller_spawn,
        
        # When the robot is spawned, load all controllers
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[
                    load_joint_state_controller,
                    rover_wheel_controller,
                    servo_controller,
                ],
            )
        ),

        gazebo,
        node_robot_state_publisher,
        spawn_entity,
    ])
