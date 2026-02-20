from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'osr_perception_challenge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.[yma]*'))),
        (os.path.join('share', package_name, 'models'), glob('models/*.pkl')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='icehot03',
    maintainer_email='pol.p.c@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    package_data={
        # Incluir los archivos de configuración y de lanzamiento
        package_name: [
            'launch/*',  # Incluir todos los archivos dentro de 'launch'
            'conf/*.yaml',  # Incluir todos los archivos YAML dentro de 'conf'
        ],
    },
    entry_points={
        'console_scripts': [
            'dar_vueltas = osr_perception_challenge.dar_vueltas:main',
            'brain_percepcion = osr_perception_challenge.brain:main',
        ],
    },
)
