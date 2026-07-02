cmake_minimum_required(VERSION 3.0.2)
project(usv_roadmap)

find_package(catkin REQUIRED COMPONENTS roscpp)
find_package(OpenCV REQUIRED COMPONENTS core imgproc ximgproc)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES usv_roadmap
  CATKIN_DEPENDS roscpp
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(usv_roadmap src/roadmap.cpp src/common.cpp)
target_link_libraries(usv_roadmap ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
