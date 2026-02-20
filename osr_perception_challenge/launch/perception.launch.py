from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():

    brain = Node(
        package='osr_perception_challenge',
        executable='brain_percepcion',
        name='brain_percepcion',
        output='screen',
        emulate_tty=True,
    )

    dar_vueltas = Node(
        package='osr_perception_challenge',
        executable='dar_vueltas',
        name='dar_vueltas',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        brain,
        TimerAction(
            period=2.0,          
            actions=[dar_vueltas]
        )
    ])
