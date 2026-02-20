import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    roboclaw_params = os.path.join(
        get_package_share_directory('osr_bringup'),
        'config',
        'roboclaw_params_mod.yaml'
    )
    
    osr_params = os.path.join(
        get_package_share_directory('osr_bringup'),
        'config',
        'osr_params_mod.yaml'
    )

    ld = LaunchDescription()
    
    ld.add_action(
        Node(
            package='osr_control',
            executable='roboclaw_wrapper',
            name='roboclaw_wrapper',
            output='screen',
            emulate_tty=True,
            respawn=True,
            parameters=[roboclaw_params]
        )
    )

    ld.add_action(
        DeclareLaunchArgument('enable_odometry', default_value='true')
    )

    ld.add_action(
        Node(
            package='osr_control',
            executable='servo_control',
            name='servo_wrapper',
            output='screen',
            emulate_tty=True,
            respawn=True,
            parameters=[{'centered_pulse_widths': [164, 148, 153, 143]}]  # pulse width where the corner motors are in their default position, see rover_bringup.md.
        )
    )

    ld.add_action(
        Node(
            package='osr_control',
            executable='rover',
            name='rover',
            output='screen',
            emulate_tty=True,
            respawn=True,
            parameters=[osr_params,
                        {'enable_odometry': LaunchConfiguration('enable_odometry')}]
        )
    )
    
    ld.add_action(
        Node(
            package='osr_control',
            executable='ina260',
            name='ina260_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {"publish_rate": 1.0},
                {"sensor_address": "0x45"},
            ]        
        )
    )

    return ld
