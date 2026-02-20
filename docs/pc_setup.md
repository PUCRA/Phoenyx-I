# 💻 PC Setup Guide  

This guide explains how to set up a **PC development environment** for Phoenyx I.  
The PC is used to:

- Run simulations in **Gazebo**
- Visualize the robot using **RViz**
- Develop and debug ROS 2 nodes
- Monitor the real robot while running

The target system is optimized for **ROS 2 Humble** running on **Ubuntu 22.04 LTS**.

---

## 📋 Requirements

### Operating System
- **Ubuntu 22.04 LTS (64-bit)**

Other Ubuntu versions are not officially supported.

### Recommended Hardware
- ≥ 16 GB RAM recommended
- Dedicated GPU recommended for Gazebo simulation
- Stable internet connection

---

## 1. System Preparation

Before installing PHOENYX-I dependencies, update the system:

```bash
sudo apt update
sudo apt upgrade -y
```

Reboot if the system was updated:

```bash
sudo reboot
```

---

## 2. Install Phoenyx I PC Environment

The complete development environment is installed using a single script.

This script automatically:

- Installs **ROS 2 Humble**
- Installs required ROS packages and system dependencies
- Installs development tools
- Creates the ROS 2 workspace
- Clones the Phoenyx I repository
- Builds the workspace
- Configures the shell environment

---

### 2.1 Download the installation script

```bash
wget https://raw.githubusercontent.com/PUCRA/Phoenyx-I/refs/heads/main/scripts/install_pc.sh
```

---

### 2.2 Make the script executable

```bash
chmod +x install_pc.sh
```

---

### 2.3 Run the installation

```bash
./install_pc.sh
```

⏳ The installation may take several minutes depending on internet speed.

---

## 3. Verify Installation

Open a new terminal and check that ROS 2 is available:

```bash
ros2 --version
```

You should see ROS 2 Humble information.

Check that the workspace was built correctly:

```bash
cd ~/phoenyxI_ws
colcon list
```

---

## 4. Test Simulation

To verify everything is working correctly, launch a simulation world:

```bash
ros2 launch osr_gazebo empty_world_simplified.launch
```

Gazebo should open with the rover model.

---

## ✅ Next Steps

Once the PC environment is ready, you can:

- Run simulations and develop new features
- Continue with the full system setup
- Connect to the real robot and visualize data in RViz

➡️ Return to the main project README:  
[`README.md`](../README.md)