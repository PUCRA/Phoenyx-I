cmake_minimum_required(VERSION 3.5)
project(unitree_lidar_ros2)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)

include_directories(
  ${PCL_INCLUDE_DIRS}
  include
  /usr/local/include
)

add_definitions(${PCL_DEFINITIONS})

add_executable(unitree_lidar_ros2_node src/unitree_lidar_ros2_node.cpp)

target_link_libraries(unitree_lidar_ros2_node
  ${PCL_LIBRARIES}
  /usr/local/lib/libunitree_lidar_sdk.a
)

ament_target_dependencies(
  unitree_lidar_ros2_node
  rclcpp 
  std_msgs 
  sensor_msgs
  geometry_msgs 
  tf2_ros
  pcl_conversions
)

# Converter PointCloud2 to LaserScan
add_executable(pointcloud_to_laserscan src/pointcloud_to_laserscan.cpp)

ament_target_dependencies(
  pointcloud_to_laserscan
  rclcpp
  sensor_msgs
)

install(TARGETS
  unitree_lidar_ros2_node
  pointcloud_to_laserscan
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files and config
install(DIRECTORY 
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install rviz file
install(FILES 
  rviz/view.rviz
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()