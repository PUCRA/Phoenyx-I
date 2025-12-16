from setuptools import find_packages, setup
import os
import glob

package_name = 'osr_control_challenge'
from setuptools import find_packages, setup
import os
import glob

package_name = 'osr_control_challenge'

# Find all YAML config files
config_files = glob.glob(os.path.join('conf', '*.yaml'))

# Find all launch files
launch_files = glob.glob(os.path.join('launch', '*.py'))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),  # Instala osr_control_challenge package
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/conf', config_files),
        ('share/' + package_name + '/launch', launch_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='judith',
    maintainer_email='judith.salvador@estudiantat.upc.edu',
    description='Control challenge for OSR maze navigation',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'maze_navigation = osr_control_challenge.maze_navigation:main',
            'killer_node = osr_control_challenge.killer_node:main',   # 👈 AÑADIDO
        ],
    },
)

