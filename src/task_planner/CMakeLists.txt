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
  scripts/core/road_network.py
  scripts/core/task_assigner.py
  scripts/core/hetero_graph.py
  scripts/core/graph_evaluator.py
  scripts/core/state_evaluator.py
  scripts/core/min_cost_flow.py
  scripts/core/domain.py
  scripts/core/route_builder.py
  scripts/algorithms/alns_scheduler.py
  scripts/algorithms/base_scheduler.py
  scripts/algorithms/scheduler.py
  scripts/algorithms/vns_scheduler.py
  scripts/algorithms/tabu_scheduler.py
  scripts/algorithms/ga_scheduler.py
  scripts/algorithms/ges_scheduler.py
  scripts/algorithms/memetic_scheduler.py
  scripts/algorithms/planning_service.py
  scripts/runtime/auction_realloc.py
  scripts/runtime/event_monitor.py
  scripts/runtime/real_time_scheduler.py
  scripts/runtime/route_executor.py
  scripts/runtime/execution_engine.py
  scripts/runtime/rolling_service.py
  scripts/runtime/task_scheduler_node.py
  scripts/visualization/visualize_road_net.py
  scripts/apps/run_demo.py
  scripts/apps/run_unified_planner.py
  scripts/apps/fleet_demo_bridge_node.py
  scripts/apps/whole_river_map_node.py
  scripts/apps/fleet_motion_sim_node.py
  scripts/apps/ais_segment_adapter_node.py
  scripts/apps/mock_local_follower_node.py
  scripts/apps/dwa_kinematic_sim_node.py
  scripts/apps/ais_dwa_tracker_node.py
  scripts/apps/local_path_viz_node.py
  scripts/apps/local_road_overlay_node.py
  scripts/learning/dataset_builder.py
  scripts/learning/train_matcher.py
  scripts/tests/test_unified_pipeline.py
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
