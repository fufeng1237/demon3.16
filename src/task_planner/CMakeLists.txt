cmake_minimum_required(VERSION 3.0.2)
project(task_planner)

add_compile_options(-std=c++17)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs ximgproc)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  grid_map_msgs
  grid_map_ros
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs nav_msgs geometry_msgs visualization_msgs grid_map_msgs grid_map_ros
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# ============================================================
# C++ Road Network Generator (from usv_roadmap_standalone)
# ============================================================
add_executable(roadmap_generator_node
  src/roadmap_generator_node.cpp
  src/common.cpp
)
target_link_libraries(roadmap_generator_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# ============================================================
# Python scripts
# ============================================================
catkin_install_python(PROGRAMS
  scripts/road_network.py
  scripts/task_assigner.py
  scripts/event_monitor.py
  scripts/task_scheduler_node.py
  scripts/visualize_road_net.py
  scripts/run_demo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
