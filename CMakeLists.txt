cmake_minimum_required(VERSION 3.0.2)
project(radarscenes2bag)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  geometry_msgs
  nav_msgs
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
  LIBRARIES radarscenes2bag
  CATKIN_DEPENDS roscpp rospy sensor_msgs geometry_msgs nav_msgs std_msgs
#  DEPENDS system_lib
)

#############
## Install ##
#############

catkin_python_setup()

catkin_install_python(PROGRAMS
  scripts/makeBagfile.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(FILES
  # myfile1
  # myfile2
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
