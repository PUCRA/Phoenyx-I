from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def _create_nodes(context, *args, **kwargs):
    
    # Get execution mode from launch argument ("sim" or "real")
    mode = LaunchConfiguration('mode').perform(context).strip().lower()
    if mode not in ('sim', 'real'):
        raise RuntimeError(f"Invalid mode '{mode}'. Use mode:=sim or mode:=real")

    pkg_share = get_package_share_directory('osr_bringup')

    # Select YAML configuration file based on mode
    yaml_name = 'joystick_sim.yaml' if mode == 'sim' else 'joystick_real.yaml'
    joy_params = os.path.join(pkg_share, 'config', yaml_name)

    # Safety check: make sure the YAML file exists
    if not os.path.exists(joy_params):
        raise RuntimeError(f"Config file not found: {joy_params}")

    # Joystick driver node
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        emulate_tty=True,
        respawn=True,
        parameters=[joy_params],
    )

    # Apply cmd_vel remapping only when running on the real robot
    # On real hardware, joystick teleoperation should use the "intuitive" velocity interface,
    # so we remap /cmd_vel to /cmd_vel_intuitive to get consistent steering behavior when reversing.
    teleop_remaps = [('/cmd_vel', '/cmd_vel_intuitive')] if mode == 'real' else []

    # Teleoperation node (converts joystick input into Twist commands)
    teleop_twist_joy_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy_node',
        output='screen',
        emulate_tty=True,
        respawn=True,
        parameters=[joy_params],
        remappings=teleop_remaps,
    )

    return [joy_node, teleop_twist_joy_node]


def generate_launch_description():
    return LaunchDescription([
        # Launch argument to choose execution mode
        DeclareLaunchArgument(
            'mode',
            default_value='real',
            description='Execution mode: sim or real'
        ),
        # Create nodes after resolving launch arguments
        OpaqueFunction(function=_create_nodes),
    ])