cmake_minimum_required(VERSION 3.0.2)
project(orb_test)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(OpenCV REQUIRED)

find_package(g2o REQUIRED)
find_package(cv_bridge REQUIRED)
include_directories("/usr/include/eigen3")
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)



catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES orb_test
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)



## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
${cv_bridge_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ./include
)

add_executable(main src/main.cpp src/orb_extractor.cpp src/vfc.cpp)
add_executable(pnp src/pnptest.cpp src/orb_extractor.cpp src/vfc.cpp)



target_link_libraries(main
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
target_link_libraries(pnp
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${cv_bridge_LIBRARIES}
  g2o::stuff g2o::core
  g2o_types_sba g2o_solver_csparse  g2o_csparse_extension g2o::types_slam2d g2o::types_slam3d
)
