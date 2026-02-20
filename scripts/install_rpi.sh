#!/bin/bash

# PUCRA – Phoenyx I (Raspberry Pi) Installation Script
# ----------------------------------------------------
# This script bootstraps a Raspberry Pi running Ubuntu Server 22.04 LTS for the Phoenyx I rover:
#  - Installs ROS 2 Humble
#  - Installs system + ROS dependencies (via rosdep)
#  - Creates a ROS 2 workspace
#  - Clones the Phoenyx repository
#  - Builds the workspace with colcon
#  - Configures the user's ~/.bashrc to source ROS and the workspace
#
# Target platform: Ubuntu Server 22.04 (aarch64) on Raspberry Pi 4
#
# Usage:
#   chmod +x install_rpi.sh
#   ./install_rpi.sh

##########################################################
#                FUNCTION DEFINITIONS                    #
##########################################################

# Install git 
install_git() {
  echo "Git is not installed. Proceeding with installation..."
  sudo apt update
  sudo apt install -y git
  echo "Git installation completed."
}

# Install ROS 2 Humble
install_ros2() {
  echo "ROS 2 Humble is not installed. Proceeding with installation..."
  
  # Set locale
  sudo apt update && sudo apt install -y locales
  sudo locale-gen en_US en_US.UTF-8
  sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
  export LANG=en_US.UTF-8
  
  # Setup sources
  sudo apt install -y software-properties-common
  sudo add-apt-repository universe -y
  sudo apt update && sudo apt install curl -y
  
  export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
  curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" # If using Ubuntu derivates use $UBUNTU_CODENAME
  sudo dpkg -i /tmp/ros2-apt-source.deb

  # Install ROS 2 packages
  sudo apt update 
  sudo apt upgrade -y
  sudo apt install ros-humble-desktop -y
  source /opt/ros/humble/setup.bash
  
  echo "ROS 2 Humble installation completed."
}

# Install the required ROS 2 packages
install_ros_packages() {
  echo "Installing the required ROS 2 packages..."
  sudo apt update

  # Needed for JPL ROS packages
  sudo apt install -y ros-humble-controller-manager
  sudo apt install -y ros-humble-robot-state-publisher
  sudo apt install -y ros-humble-joint-state-publisher
  sudo apt install -y ros-humble-joint-state-publisher-gui 
  sudo apt install -y ros-humble-trajectory-msgs
  sudo apt install -y ros-humble-velocity-controllers
  sudo apt install -y ros-humble-joint-trajectory-controller

  # Other packages needed
  sudo apt install -y ros-humble-nav2-msgs
  sudo apt install -y ros-humble-joy-tester
  sudo apt install -y ros-humble-xacro
  sudo apt install -y ros-humble-rplidar-ros

  echo "ROS 2 packages intallation completed."
}

# Install colcon common extensions
install_colcon_extensions() {
  sudo apt update
  sudo apt install -y python3-colcon-common-extensions
  echo "colcon common extensions installation completed."
}

# Install rosdep + pip prereqs
install_python_and_rosdep_tools() {
  echo "Installing python3-pip and rosdep..."
  sudo apt update
  sudo apt install -y python3-pip python3-rosdep

  # rosdep init only once system-wide
  if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    echo "Initializing rosdep..."
    sudo rosdep init
  else
    echo "rosdep already initialized."
  fi

  echo "Updating rosdep database..."
  rosdep update
}

# Install pip deps for hardware libs
install_hardware_python_deps() {
  echo "Installing hardware Python dependencies (user-local)..."
  python3 -m pip install --user --upgrade pip

  # These are used for IMU, servos, INA260, GPIO, and Roboclaw motor controller
  python3 -m pip install --user \
    adafruit-circuitpython-blinka \
    adafruit-circuitpython-mpu6050 \
    adafruit-circuitpython-servokit \
    adafruit-circuitpython-ina260 \
    ina260 \
    RPi.GPIO \
    smbus  \
    roboclaw 
}

install_pycrc_from_github() {
  echo "Installing PyCRC from GitHub (alexbutirskiy/PyCRC)..."

  local PYCRC_DIR="$HOME/.local/src/PyCRC"
  local PYCRC_REPO="https://github.com/alexbutirskiy/PyCRC.git"

  sudo apt update
  sudo apt install -y git python3-pip

  mkdir -p "$HOME/.local/src"

  if [ -d "${PYCRC_DIR}/.git" ]; then
    echo "PyCRC repo already exists. Updating..."
    git -C "$PYCRC_DIR" pull --ff-only
  else
    git clone "$PYCRC_REPO" "$PYCRC_DIR"
  fi

  # Install using pip from local path (user-local, safer than sudo pip)
  python3 -m pip install --user --upgrade pip
  python3 -m pip install --user "$PYCRC_DIR"

  echo "PyCRC installed successfully."
}

install_realsense_deps() {
  echo "Installing Intel RealSense Python dependencies (user-local)..."

  python3 pip install pyrealsense2 # RealSense Python bindings 
  python3 -m pip install joblib # ML deps you listed
  python3 -m pip install scikit-learn
  python3 -m pip install --upgrade opencv-python
  python3 -m pip install --upgrade scipy scikit-learn joblib
  python3 -m pip install --user --force-reinstall "scikit-learn==1.6.1" joblib
  python3 -m pip install --user --force-reinstall "numpy<2"
  
  echo "RealSense dependencies installed."
}

# Configurate the bash file
configure_bashrc() {
  echo "Configuring the bash file..."
  # Ensure ROS 2 Humble environment setup is sourced in .bashrc
  if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    echo "Adding ROS 2 Humble environment setup to .bashrc"
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
  fi

  # Add colcon argcomplete to bashrc
  if ! grep -q "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" ~/.bashrc; then
    echo "Adding colcon argcomplete to .bashrc"
    echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc
  fi
   
  # Add workspace setup to bashrc
  if ! grep -q "source ~/phoenyxI_ws/install/setup.bash" ~/.bashrc; then
    echo "Adding ROS 2 workspace setup to .bashrc"
    echo "source ~/phoenyxI_ws/install/setup.bash" >> ~/.bashrc
  fi
  
  # Comment any other workspace setup lines
  sed -i '/source ~\/.*\/install\/setup.bash/ { /source ~\/phoenyxI_ws\/install\/setup.bash/!s/^/#/ }' ~/.bashrc
  
  # Comment any other doamain id lines
  sed -i '/export ROS_DOMAIN_ID=/ { /export ROS_DOMAIN_ID=58/!s/^/#/ }' ~/.bashrc
 
  # Add network configuration to bashrc
  if ! grep -q "export ROS_DOMAIN_ID=58" ~/.bashrc; then
    echo "Adding export ROS_DOMAIN_ID"
    echo "export ROS_DOMAIN_ID=58" >> ~/.bashrc
  fi
  
    
  # Source the updated .bashrc
  source ~/.bashrc
  
  echo "Bash file configuration finished"
}

# Create ROS 2 workspace
create_ros2_workspace() {
  # Create ROS 2 workspace if it doesn't exist
  echo "Creating ROS 2 workspace..."
  mkdir -p ~/phoenyxI_ws/src
  echo "ROS 2 workspace created."
}

# Clone this repository into the ROS 2 workspace
clone_phoenyxI_repository() {
  echo "Cloning Phoenyx-I repository into the ROS 2 workspace..."
  cd ~/phoenyxI_ws/src/
  git clone https://github.com/PUCRA/Phoenyx-I.git . 
  echo "Phoenyx-I repository cloned successfully."
}

install_udev_rules() {
  echo "Installing udev rules for Phoenyx I..."

  UDEV_SRC="$HOME/phoenyxI_ws/src/config"
  UDEV_DST="/etc/udev/rules.d"

  if [ ! -d "$UDEV_SRC" ]; then
    echo "⚠️  udev rules directory not found: $UDEV_SRC"
    echo "Skipping udev rules installation."
    return
  fi

  sudo cp "$UDEV_SRC"/* "$UDEV_DST"/

  sudo udevadm control --reload-rules
  sudo udevadm trigger

  echo "udev rules installed successfully."
}

# rosdep install for workspace
install_workspace_deps_with_rosdep() {
  echo "Installing workspace dependencies with rosdep..."
  # Ensure ROS env is available in this shell
  source /opt/ros/humble/setup.bash

  cd "${WORKSPACE}"
  rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
}

# Build ROS 2 workspace
build_ros2_workspace() {
  echo "Building ROS 2 workspace..."
  cd ~/phoenyxI_ws/
  colcon build
  echo "ROS 2 workspace build completed."
  
  # Source the updated .bashrc
  source ~/.bashrc
}


##########################################################
#                        MAIN                            #
##########################################################

echo "Installing the necesary resources..."

# Check if Git is install
if ! command -v git &> /dev/null; then
  install_git
else
  echo "Git is already installed."
fi

# Check if ROS 2 Humble is installed
if ! dpkg -l | grep -q ros-humble; then
  install_ros2
else
  echo "ROS 2 Humble is already installed."
fi

# Installing the required ROS 2 packages
install_ros_packages

# Check if pyhton colcon extensions are installed
if ! dpkg -l | grep -q python3-colcon-common-extensions; then
   install_colcon_extensions
else
   echo "colcon common extensions are already installed."
fi

install_python_and_rosdep_tools

create_ros2_workspace
clone_phoenyxI_repository

install_udev_rules
install_workspace_deps_with_rosdep
install_hardware_python_deps
install_pycrc_from_github
install_realsense_deps

build_ros2_workspace
configure_bashrc

echo ""
echo "✅ Phoenyx I environment installed successfully!"
echo "📂 Workspace located at ~/phoenyxI_ws"
echo "💡 Run 'source ~/.bashrc' or open a new terminal to use ROS 2 commands."
