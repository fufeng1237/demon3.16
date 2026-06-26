cmake_minimum_required(VERSION 3.0.2)
project(ais_navigation)

## Compile as C++14
add_compile_options(-std=c++14)
set(Eigen3_DIR "/usr/include/eigen3")

## Find catkin macros and libraries
find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  grid_map_msgs
  grid_map_ros
  roscpp
  std_msgs
  nav_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS grid_map_msgs grid_map_ros roscpp std_msgs nav_msgs geometry_msgs
)

###########
## Build ##
###########

set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
include_directories(
  BEFORE SYSTEM /usr/include/eigen3
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_definitions(-DEIGEN_NO_DEBUG)
add_definitions(-DEIGEN_NO_STATIC_ASSERT)

# AIS lane-guided A* planner node
add_executable(ais_planner_node src/ais_planner_node.cpp)
target_link_libraries(ais_planner_node ${catkin_LIBRARIES})

# AIS lane-constrained path optimizer node
add_executable(ais_path_optimizer_node src/ais_path_optimizer_node.cpp)
target_link_libraries(ais_path_optimizer_node ${catkin_LIBRARIES})

#############
## Install ##
#############

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/generate_navigation_field.py
  scripts/visualize_navigation_field.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

# Install map files
install(DIRECTORY map/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/map
)
