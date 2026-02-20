#!/bin/bash

# PUCRA – Phoenyx I (Raspberry Pi) Installation Script
# ----------------------------------------------------
# This script bootstraps a PC running Ubuntu 22.04 LTS for the Phoenyx I rover development environment:
#  - Installs Visual Studio Code, terminator and git
#  - Installs ROS 2 Humble and the required ROS 2 packages
#  - Creates a ROS 2 workspace
#  - Clones the Phoenyx-I repository
#  - Builds the workspace with colcon
#  - Configures the user's ~/.bashrc to source ROS and the workspace
#
# Target platform: Ubuntu 22.04 LTS (amd64) on a development PC/laptop
#
# Usage:
#   chmod +x install_pc.sh
#   ./install_pc.sh

##########################################################
#                FUNCTION DEFINITIONS                    #
##########################################################

# Install Visual Studio Code 
install_vscode() {
  echo "Visual Studio Code is not installed. Proceeding with installation..."
  sudo apt update 
  sudo snap install --classic code
  echo "Visual Studio Code installation completed."
}

# Install terminator
install_terminator() {
  echo "Terminator is not installed. Proceeding with installation..."
  sudo apt update 
  sudo apt install -y terminator
  echo "Terminator installation completed."
}

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

  sudo apt install -y ros-humble-xacro
  sudo apt install -y ros-humble-gazebo-ros
  sudo apt install -y ros-humble-gazebo-ros-pkgs
  sudo apt install -y ros-humble-rviz2

  # Needed for JPL ROS packages
  sudo apt install -y ros-humble-controller-manager
  sudo apt install -y ros-humble-robot-state-publisher
  sudo apt install -y ros-humble-joint-state-publisher
  sudo apt install -y ros-humble-joint-state-publisher-gui 
  sudo apt install -y ros-humble-trajectory-msgs
  sudo apt install -y ros-humble-velocity-controllers
  sudo apt install -y ros-humble-joint-trajectory-controller
  sudo apt install -y ros-humble-gazebo-ros2-control-demos
  sudo apt install -y ros-humble-urdf-tutorial

  # Other packages needed
  sudo apt install -y ros-humble-nav2-msgs
  sudo apt install -y ros-humble-joy-tester

  echo "ROS 2 packages intallation completed."
}

# Install colcon common extensions
install_colcon_extensions() {
  sudo apt update
  sudo apt install -y python3-colcon-common-extensions
  echo "colcon common extensions installation completed."
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

  # Add gazebo to bashrc
  if ! grep -q "source /usr/share/gazebo/setup.bash" ~/.bashrc; then
    echo "Adding gazebo to .bashrc"
    echo "source /usr/share/gazebo/setup.bash" >> ~/.bashrc
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

# Remove COLCON_IGNORE file to enable osr_gazebo package on PC
remove_colcon_ignore_gazebo() {
  echo "Checking for COLCON_IGNORE in osr_gazebo package..."

  COLCON_IGNORE_PATH="$HOME/phoenyxI_ws/src/osr_gazebo/COLCON_IGNORE.txt"

  if [ -f "$COLCON_IGNORE_PATH" ]; then
    echo "Removing COLCON_IGNORE.txt from osr_gazebo (PC build enabled)"
    rm "$COLCON_IGNORE_PATH"
  else
    echo "No COLCON_IGNORE.txt found in osr_gazebo. Nothing to do."
  fi
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


# Check if Visual Studio Code is installed
if ! snap list | grep -q code; then
  install_vscode
else
  echo "Visual Studio Code is already installed."
fi

# Check if terminator is installed
if ! dpkg -l | grep -q terminator; then
  install_terminator
else
  echo "Terminator is already installed."
fi

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

create_ros2_workspace
clone_phoenyxI_repository

remove_colcon_ignore_gazebo

install_realsense_deps

build_ros2_workspace
configure_bashrc

echo ""
echo "✅ Phoenyx I environment installed successfully!"
echo "📂 Workspace located at ~/phoenyxI_ws"
echo "💡 Run 'source ~/.bashrc' or open a new terminal to use ROS 2 commands."
