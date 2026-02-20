import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    perception_dir = get_package_share_directory('osr_perception_challenge')

    perception_launch = os.path.join(
        perception_dir,
        'launch',
        'perception.launch.py'
    )

    # --- Include perception launch (no delay) ---
    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(perception_launch)
    )

    return LaunchDescription([
        LogInfo(msg='[osr_bringup] Starting OSR bringup...'),
        perception
    ])
