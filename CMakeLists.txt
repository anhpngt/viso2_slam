cmake_minimum_required(VERSION 2.8.3)
project(viso2_slam)

find_package(catkin REQUIRED COMPONENTS 
  roscpp
  sensor_msgs 
  nav_msgs 
  message_filters 
  image_transport 
  cv_bridge 
  image_geometry 
  tf 
  pcl_ros 
  std_srvs
  message_generation
)

find_package(OpenCV REQUIRED QUIET)
find_package(PCL REQUIRED QUIET)
find_package(Boost REQUIRED QUIET COMPONENTS signals thread)

add_message_files(DIRECTORY msg FILES VisoInfo.msg)

generate_messages(
  DEPENDENCIES std_msgs
)

catkin_package(
  INCLUDE_DIRS include
)

SET(CMAKE_CXX_FLAGS "-std=c++11 -msse3 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS} 
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

add_library(libviso2
  include/filter.cpp
  include/matcher.cpp
  include/matrix.cpp
  include/reconstruction.cpp
  include/triangle.cpp
  include/viso.cpp
  include/viso_mono.cpp
  include/viso_stereo.cpp
)
  
add_executable(stereo_slam src/stereo_odometer.cpp)
add_dependencies(stereo_slam 
  ${catkin_EXPORTED_TARGETS} 
  ${PROJECT_NAME}_gencpp
)
target_link_libraries(stereo_slam
  libviso2
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
)