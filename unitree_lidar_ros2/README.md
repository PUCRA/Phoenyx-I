# 📡 Unitree LiDAR ROS2 – README

# ⚪ lidar_full.launch.py

ROS2 launch file that initializes the Unitree LiDAR node and visualizes the data in real time using RViz2.

---

## 🚀 Nodes launched

**`unitree_lidar_ros2_node`**
Main node that handles communication with the Unitree LiDAR and publishes point cloud and IMU data.

**`rviz2`**
Visualizer that automatically loads the `view.rviz` configuration included in the package.

---

## ⚙️ Configuration Parameters

| Parameter | Default Value | Description |
|---|---|---|
| `initialize_type` | `2` | LiDAR initialization type |
| `work_mode` | `0` | Work mode |
| `use_system_timestamp` | `True` | Use system timestamp |
| `range_min` | `0.0` | Minimum detection range (m) |
| `range_max` | `100.0` | Maximum detection range (m) |
| `cloud_scan_num` | `18` | Number of scans per point cloud |
| `serial_port` | `/dev/ttyACM0` | LiDAR serial port |
| `baudrate` | `2000000` | Serial communication baudrate |
| `lidar_port` | `6101` | LiDAR UDP port |
| `lidar_ip` | `192.168.1.62` | LiDAR IP address |
| `local_port` | `6201` | Local UDP port |
| `local_ip` | `192.168.1.100` | Local IP address |
| `cloud_frame` | `unilidar_lidar` | Point cloud frame |
| `cloud_topic` | `unilidar/cloud` | Point cloud topic |
| `imu_frame` | `unilidar_imu` | IMU frame |
| `imu_topic` | `unilidar/imu` | IMU topic |

---

## 📡 Published Topics

| Topic | Type | Description |
|---|---|---|
| `unilidar/cloud` | `sensor_msgs/PointCloud2` | LiDAR point cloud |
| `unilidar/imu` | `sensor_msgs/Imu` | Integrated IMU data |

---

## 🖥️ Usage
```bash
ros2 launch unitree_lidar_ros2 lidar_launch.py
```

---

## 📋 Requirements

- ROS2 (Humble or later)
- `unitree_lidar_ros2` package installed
- RViz2 installed
- Unitree LiDAR connected via `/dev/ttyACM0` or over the network at `192.168.1.62`

---

## 📝 Notes

> ⚠️ The local IP (`192.168.1.100`) and the LiDAR IP (`192.168.1.62`) must be on the same subnet.

> 💡 If the LiDAR is connected via USB instead of network, verify the correct port with:
> ```bash
> ls /dev/ttyACM*
> ```

> 📄 The RViz configuration is loaded automatically from the package's `share` directory.

&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;






# ⚪ cloud_to_laser.launch.py

ROS2 launch file that initializes the Unitree LiDAR node, converts the point cloud into a 2D laser scan, and optionally visualizes the data in RViz2.

---

## 🚀 Nodes launched

**`unitree_lidar_ros2_node`**
Main node that handles communication with the Unitree LiDAR and publishes point cloud and IMU data.

**`pointcloud_to_laserscan`**
Converter node that transforms the 3D `PointCloud2` data into a 2D `LaserScan` message. It can be configured via a `.yaml` file or with hardcoded default parameters.

**`rviz2`** *(optional)*
Visualizer that automatically loads the `view.rviz` configuration included in the package. If RViz2 is not available, the launch file will still run the LiDAR and converter nodes.

---

## ⚙️ LiDAR Parameters

| Parameter | Default Value | Description |
|---|---|---|
| `initialize_type` | `2` | LiDAR initialization type |
| `work_mode` | `0` | Work mode |
| `use_system_timestamp` | `True` | Use system timestamp |
| `range_min` | `0.0` | Minimum detection range (m) |
| `range_max` | `100.0` | Maximum detection range (m) |
| `cloud_scan_num` | `18` | Number of scans per point cloud |
| `serial_port` | `/dev/ttyACM0` | LiDAR serial port |
| `baudrate` | `2000000` | Serial communication baudrate |
| `lidar_port` | `6101` | LiDAR UDP port |
| `lidar_ip` | `192.168.1.62` | LiDAR IP address |
| `local_port` | `6201` | Local UDP port |
| `local_ip` | `192.168.1.100` | Local IP address |
| `cloud_frame` | `unilidar_lidar` | Point cloud frame |
| `cloud_topic` | `unilidar/cloud` | Point cloud topic |
| `imu_frame` | `unilidar_imu` | IMU frame |
| `imu_topic` | `unilidar/imu` | IMU topic |

---

## 🔄 Converter Parameters (PointCloud2 → LaserScan)

| Parameter | Default Value | Description |
|---|---|---|
| `min_height` | `-0.5` | Minimum height slice (m) |
| `max_height` | `0.5` | Maximum height slice (m) |
| `angle_min` | `-π` | Start angle of the scan (rad) |
| `angle_max` | `+π` | End angle of the scan (rad) |
| `angle_increment` | `0.004363` | Angular resolution – 0.25° (rad) |
| `range_min` | `0.05` | Minimum valid range (m) |
| `range_max` | `10.0` | Maximum valid range (m) |
| `rotation_offset` | `0.0` | Rotation offset if LiDAR is physically rotated (rad) |
| `target_frame` | `unilidar_lidar` | Target TF frame for the laser scan |

> 💡 If a `pointcloud_to_laserscan_params.yaml` config file is found in the package's `config/` directory, it will be used automatically instead of the default parameters above.

---

## 📡 Topics

| Topic | Type | Description |
|---|---|---|
| `unilidar/cloud` | `sensor_msgs/PointCloud2` | 3D LiDAR point cloud |
| `unilidar/imu` | `sensor_msgs/Imu` | Integrated IMU data |
| `scan` | `sensor_msgs/LaserScan` | Converted 2D laser scan |

---

## 🖥️ Usage
```bash
ros2 launch unitree_lidar_ros2 cloud_to_laser.launch.py
```

---

## 📋 Requirements

- ROS2 (Humble or later)
- `unitree_lidar_ros2` package installed with the `pointcloud_to_laserscan` executable
- RViz2 installed *(optional)*
- Unitree LiDAR connected via `/dev/ttyACM0` or over the network at `192.168.1.62`

---

## 📝 Notes

> ⚠️ The local IP (`192.168.1.100`) and the LiDAR IP (`192.168.1.62`) must be on the same subnet.

> 💡 If the LiDAR is connected via USB instead of network, verify the correct port with:
> ```bash
> ls /dev/ttyACM*
> ```

> 📄 If `pointcloud_to_laserscan_params.yaml` exists in `config/`, it takes priority over the default hardcoded parameters.

> ✅ If RViz2 is not available, the launch file will still run successfully with just the LiDAR and converter nodes.


&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;




# ⚪ pointcloud_to_laserscan.cpp

ROS2 C++ node that subscribes to a 3D `PointCloud2` topic and converts it into a 2D `LaserScan` message by slicing the point cloud at a configurable height range.

---

## 🧠 How it works

1. Subscribes to `/unilidar/cloud` (`PointCloud2`)
2. Filters points by **height** (Z axis) to extract a horizontal plane slice
3. For each valid point, calculates its **angle and distance** in the XY plane
4. Assigns each point to a **beam index** and keeps the **closest distance** per beam
5. Publishes the resulting 2D scan to `/scan` (`LaserScan`)

---

## ⚙️ Parameters

| Parameter | Default Value | Description |
|---|---|---|
| `min_height` | `0.0` | Minimum Z height of the plane slice (m) |
| `max_height` | `0.2` | Maximum Z height of the plane slice (m) |
| `angle_min` | `-π` | Start angle of the scan (rad) |
| `angle_max` | `+π` | End angle of the scan (rad) |
| `angle_increment` | `0.004363` | Angular resolution – 0.25° (rad) |
| `range_min` | `0.05` | Minimum valid range (m) |
| `range_max` | `30.0` | Maximum valid range (m) |
| `rotation_offset` | `0.0` | Rotation offset if LiDAR is physically rotated (rad) |
| `target_frame` | `base_link` | TF frame for the output LaserScan |
| `outlier_threshold` | `5.0` | Distance threshold for outlier filtering (m) |

---

## 📡 Topics

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/unilidar/cloud` | `sensor_msgs/PointCloud2` | Subscribed | Input 3D point cloud |
| `/scan` | `sensor_msgs/LaserScan` | Published | Output 2D laser scan |

---

## 🔍 Point filtering pipeline
```
PointCloud2
    │
    ├─ ❌ Discard NaN points
    ├─ ❌ Discard points outside height range [min_height, max_height]
    ├─ ❌ Discard points outside distance range [range_min, range_max]
    ├─ ❌ Discard points outside angular range [angle_min, angle_max]
    │
    └─ ✅ Assign to beam → keep closest distance per beam
    │
    └─► Publish /scan
```

---

## 📊 Logging

Every **30 messages**, the node logs a summary to the console:
```
✅ Total: 12400 | En plano [0.00,0.20]m: 340 (2.7%) | Beams válidos: 312/1441 | Min dist: 0.43m
```

| Field | Description |
|---|---|
| `Total` | Total points received in the cloud |
| `En plano` | Points that passed the height filter |
| `Beams válidos` | Beams with at least one valid detection |
| `Min dist` | Closest detected obstacle |

---

## 🖥️ Usage

This node is included in the `unitree_lidar_ros2` package and can be launched directly or via a launch file:
```bash
# Via launch file
ros2 launch unitree_lidar_ros2 cloud_to_laser.launch.py

# Standalone
ros2 run unitree_lidar_ros2 pointcloud_to_laserscan
```

To override parameters at runtime:
```bash
ros2 run unitree_lidar_ros2 pointcloud_to_laserscan \
  --ros-args \
  -p min_height:=-0.1 \
  -p max_height:=0.1 \
  -p range_max:=15.0 \
  -p target_frame:=unilidar_lidar
```

---

## 📋 Requirements

- ROS2 (Humble or later)
- `sensor_msgs` package
- `rclcpp` package
- Unitree LiDAR publishing on `/unilidar/cloud`

---

## 📝 Notes

> ⚠️ The `target_frame` must match the TF tree of your robot. Use `unilidar_lidar` if no TF is configured yet.

> 💡 Tune `min_height` and `max_height` carefully — the thickness of the slice directly affects how many points are captured and the quality of the resulting scan.

> 📄 The node keeps the **minimum distance per beam**, meaning it always reports the closest obstacle detected within the height slice for each angle.


&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;



# ⚪ unitree_lidar_ros2_node.cpp

ROS2 C++ entry point that initializes and spins the `UnitreeLidarSDKNode`, the main node responsible for interfacing with the Unitree LiDAR hardware.

---

## 🧠 How it works

This file is the **main entry point** of the Unitree LiDAR ROS2 driver. It does not contain any logic itself — it simply:

1. Initializes the ROS2 runtime (`rclcpp::init`)
2. Instantiates the `UnitreeLidarSDKNode` defined in `unitree_lidar_ros2.h`
3. Spins the node to keep it alive and processing callbacks
4. Shuts down cleanly on exit

All LiDAR logic, parameters, and topic publishing are handled inside `UnitreeLidarSDKNode`.

---

## 📡 Topics published

> Defined in `unitree_lidar_ros2.h` — listed here for reference.

| Topic | Type | Description |
|---|---|---|
| `unilidar/cloud` | `sensor_msgs/PointCloud2` | 3D LiDAR point cloud |
| `unilidar/imu` | `sensor_msgs/Imu` | Integrated IMU data |

---

## 🖥️ Usage

This node is built as part of the `unitree_lidar_ros2` package and can be run directly or via a launch file:
```bash
# Standalone
ros2 run unitree_lidar_ros2 unitree_lidar_ros2_node

# Via launch file (recommended)
ros2 launch unitree_lidar_ros2 lidar_launch.py
```

---

## 📋 Requirements

- ROS2 (Humble or later)
- `rclcpp` package
- `unitree_lidar_ros2.h` and its dependencies compiled and available
- Unitree LiDAR hardware connected via USB (`/dev/ttyACM0`) or network (`192.168.1.62`)

---

## 📝 Notes

> 📄 This file only contains the `main()` function. All node logic is implemented in `unitree_lidar_ros2.h`.

> ⚠️ Do not modify this file unless you need to change node initialization options. Configuration is done via ROS2 parameters at launch time.

> 💡 Copyright © 2020-2024 Unitree Robotics Co. Ltd. All rights reserved.