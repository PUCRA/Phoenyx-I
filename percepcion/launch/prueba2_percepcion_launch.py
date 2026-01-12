import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import yaml

def load_yaml(file_path, namespace):
    """Carga solo los parámetros de un namespace específico de un archivo YAML."""
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data.get(namespace, {}).get("ros__parameters", {})
    except Exception as e:
        print(f"Error cargando YAML: {e}")
        return {}

def generate_launch_description():

    # osr_bringup_dir = get_package_share_directory('osr_bringup')
    # osr_mod_launch = os.path.join(osr_bringup_dir, 'launch', 'osr_mod_launch.py')

    # percepcion_config = os.path.join(
    #     get_package_share_directory('percepcion'),
    #     'config',
    #     'lectura_cam.yaml'
    # )

    # bringup = os.path.join(
    #     get_package_share_directory('osr_bringup'),
    #     'launch',
    #     'osr_mod_launch.py'
    # )
    return LaunchDescription([
        # Incluir el launch de osr_mod_launch
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(osr_mod_launch)
        # ),
        # Nodos adicionales no incluidos en osr_mod_launch
        # IncludeLaunchDescription(
            # PythonLaunchDescriptionSource(bringup)
        # ),
        # Node(
        #     package='percepcion',
        #     executable='dar_vueltas',
        #     name='dar_vueltas',
        #     output='screen',
        #     emulate_tty=True,
        # ),
        Node(
            package='percepcion',
            executable='brain_percepcion',
            name='brain_percepcion',
            output='screen',
            emulate_tty=True,
        )
    ])
