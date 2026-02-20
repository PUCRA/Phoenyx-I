<p align="center">
  <img src="docs/images/PHOENYX-I-Banner.png" alt="Project Logo"/>
</p>

> ### **🚀 From Mars to the Lab:**
>
> 🤖 **Meet PHOENYX-I**, an autonomous exploration rover inspired by **NASA’s designs**, bringing **cutting-edge robotics** from the lab to Mars-like terrains.
>
> 🧠 Built on **ROS 2**, Phoenyx I combines **real-time perception**, **LiDAR-based navigation**, and **visual localization**, enabling fully autonomous operation in both simulation and real-world environments.
> 
> 🏆 **Winner of Best Overall Rover & Design Excellence** at **Sener-CEA's Bot Talent competition**.

## 👀 Watch PHOENYX-I in action 
 
<p align="center">

[![Phoenyx I – Autonomous Rover Demo](https://img.youtube.com/vi/FE0RjScWbhM/maxresdefault.jpg)](https://www.youtube.com/watch?v=FE0RjScWbhM)<br>
</p>

| 🔍 Perception Task | 🛣️ Control Task | 📍 Guidance Task |
|---|---|---|
| [![Perception](https://img.youtube.com/vi/KKVDeQCz0RY/maxresdefault.jpg)](https://youtu.be/KKVDeQCz0RY) | [![Control](https://img.youtube.com/vi/z-JHUpsCd6w/maxresdefault.jpg)](https://youtu.be/z-JHUpsCd6w)  | [![Guidance](https://img.youtube.com/vi/otcly1f2bjU/maxresdefault.jpg)](https://youtu.be/otcly1f2bjU) |
| <p align="center">Real-time digit and color recognition.</p> | <p align="center">Autonomous corridor navigation using only 2D LiDAR.</p> | <p align="center">Localization and waypoint navigation using ArUco markers.</p> |

<p align="center"><sub>▶️ Click any image to watch the corresponding video on YouTube</sub></p>


---

## 🧭 Project Overview

**PHOENYX-I** is an award-winning autonomous rover developed by undergraduate students from **PUCRA**, the robotics association at the **Polytechnic University of Catalonia**.

This repository contains the complete **ROS 2 autonomy stack** used on the robot, including simulation environments, perception pipelines, control nodes and navigation modules deployed both in competition and real-world operation.

Built on top of the [NASA JPL Open Source Rover](https://github.com/nasa-jpl/open-source-rover), **PHOENYX-I** extends the base platform with a modular autonomy architecture optimized for embedded computing on a **Raspberry Pi 4B (4 GB RAM)**.

The system integrates multiple sensors to enable autonomous operation:

- 🎥 **Intel RealSense D435i** — RGB-D perception and visual detection  
- 📡 **RPLidar** — 2D LiDAR for navigation and obstacle detection  
- 🧭 **MPU6050 IMU** — orientation estimation and motion stabilization  
- 🔋 **INA260** — power monitoring and battery supervision 

✨ PHOENYX-I is featured in the official [Open Source Rover repository](https://github.com/nasa-jpl/open-source-rover/tree/master/examples) as a community example project. We are very proud to contribute back to the same open-source initiative that originally inspired and enabled this project.

<p align="center">
  <img src="https://img.shields.io/badge/ROS2-Humble-blue"/>
  <img src="https://img.shields.io/badge/Ubuntu-22.04-orange"/>
  <img src="https://img.shields.io/badge/Hardware-RaspberryPi4B-red"/>
  <img src="https://img.shields.io/badge/License-MIT-green"/>
</p>

## 📦 Jump to:

- [🚀 Quick Start ](#-quick-start)
- [🚦 How to Run](#-how-to-run-the-system) 
- [📂 Repo Structure](#-repository-structure)
- [🏁 Competition Results](#-competition-results)
- [🤝 Want to Collaborate?](#-want-to-collaborate)
- [🌐 Join & Follow Us](#-join--follow-us)

## 🚀 Quick Start 

### 💻 PC or Laptop

The PC environment is used for:

- Running simulations in **Gazebo**
- Visualizing and monitoring the robot using **RViz**
- Development and debugging of ROS 2 nodes

👉 **Follow the complete PC setup guide here:**  
➡️ [`docs/pc_setup.md`](docs/pc_setup.md)

### 🍓 Raspberry Pi

To run Phoenyx I on the real robot, you first need to set up the onboard Raspberry Pi.

👉 **Follow the complete Raspberry Pi setup guide here:**  
➡️ [`docs/raspberry_pi_setup.md`](docs/raspberry_pi_setup.md)

### ⚙️ Steering Servo Calibration

Before operating the rover on real hardware, the steering servos **must be calibrated** to ensure all wheels are correctly aligned and the full steering range is used.

This procedure is **only required once**, or whenever a steering servo or corner assembly is replaced.

👉 **Follow the complete steering servo calibration guide here:**  
➡️ [`docs/steering_servo_calibration.md`](docs/steering_servo_calibration.md)
  

## 🚦 How to Run the System

### 🧪 In Simulation 

#### Empty Worlds

Launch an empty simulation world with the rover model:
```bash
ros2 launch osr_gazebo empty_world_simplified.launch.py
```

#### 🎮 Move With a Controller

Once Gazebo is running, you can control the rover manually:

```bash
ros2 launch osr_bringup joystick.launch.py mode:=sim
```

> [!NOTE]
> For details on the controller layout, how to change the joystick mapping, and a deeper explanation of how the teleoperation pipeline works, see: [Joystick Teleoperation Documentation](docs/joystick_teleoperation.md)

#### 🛣️ Control Task (Maze Navigation)

This mode evaluates the rover’s autonomous control inside maze environments.

Available maze worlds:

- `maze_1.world` — basic layout (recommended for tuning)
- `maze_2.world` — intermediate difficulty
- `maze_3.world` — advanced maze

RViz visualization can be enabled or disabled:
- `rviz:=true`  — launch RViz with navigation visualization
- `rviz:=false` — run simulation without RViz (lower CPU usage)

```bash
ros2 launch osr_control_challenge maze_navigation.launch.py maze:=maze_1.world rviz:=true
```

---

### 🤖 On Real Robot 

⚠️ Always start the robot bringup first in a separate terminal.

Terminal 1 — Robot Bringup (Required for all tasks)
```bash
ros2 launch osr_bringup osr_mod_launch.py
```

#### 🎮 Move With a Controller

Terminal 2
```bash
ros2 launch osr_bringup joystick.launch.py mode:=real
```

#### 🔍 Perception task

Terminal 2
``` bash
ros2 launch osr_bringup perception_challenge.launch.py 
```

#### 🛣️ Control task

Terminal 2
``` bash
ros2 launch osr_bringup control_challenge.launch.py 
```

#### 📍 Guidance task 

Terminal 2
```bash
ros2 launch guiado guiado.launch.py
```

> [!note]
> Press A on the joystick to start autonomous mode `/joy` topic.

## 📁 Repository Structure

The repository is organized following a modular ROS 2 architecture, where each package represents a specific subsystem of the rover.

```bash
├── src/
│   ├── osr_bringup/    # Robot bringup, main launch files
│   ├── osr_control/    # Base control nodes and low-level motion interfaces
│   ├── osr_control_challenge/  # Autonomous control task 
│   ├── osr_guidance_challenge/ # SLAM-based localization and waypoint navigation
│   ├── osr_perception_challenge/ # Vision-based perception (digit and color recognition)
│   ├── osr_gazebo/     # Gazebo simulation environments and worlds
│   ├── osr_interfaces/ # Custom ROS 2 messages and interfaces
│   
│   ├── config/         # udev rules (Raspberry Pi setup)
│   ├── docs/           # Project documentation and setup guides
│   ├── scripts/        # Installation and automation scripts
│
├── LICENSE.md
├── JPL_NASA_LICENCE.txt
└── README.md
```

## 🏁 Competition Results

- 🥇 **First Place Overall – Bot Talent 2025**  
- 🧠 **Awarded for Best Robot Design**
  
## License

This main project, including all contributions by PUCRA and its collaborators, is licensed under the **[MIT License](LICENSE)**.

Parts of this project derived from the work of the **Jet Propulsion Laboratory (JPL) of NASA** are covered by the **[Apache License, Version 2.0](JPL_NASA_License.txt)**. Please refer to the `JPL_NASA_License.txt` file for full terms and original attributions.

## 🤝 Want to Collaborate?

If you're interested in contributing code, improving documentation, or developing new features, feel free to check our [CONTRIBUTING](.github/CONTRIBUTING.md) page!

## 🌐 Join & Follow Us

Stay connected with PUCRA and follow our journey:

[![email](https://img.shields.io/badge/Email-D14836?logo=gmail&logoColor=white)](mailto:pucra.eebe@upc.edu) 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/pucra-upcc/posts/?feedView=all)
[![Instagram](https://img.shields.io/badge/Instagram-%23E4405F.svg?logo=Instagram&logoColor=white)](https://www.instagram.com/pucra.upc/)
[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?logo=YouTube&logoColor=white)](https://www.youtube.com/@pucraupc) 

<p align="center">
  <img src="docs/images/PHOENYX-I-Logo.png" alt="Project Logo"/>
</p>
