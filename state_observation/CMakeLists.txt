cmake_minimum_required(VERSION 3.5)
project(state_observation)

# Find dependencies
# Find dependencies
foreach(PACKAGE 
  ament_cmake 
  ament_cmake_python
  rclcpp
  rclpy 
  geometry_msgs
  nav_msgs
  ackermann_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  Eigen3
  tf2_ros
  utilities
)

  find_package(${PACKAGE} REQUIRED)

endforeach()

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install entry points
install(PROGRAMS
  state_observation/state_observation_node.py
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()