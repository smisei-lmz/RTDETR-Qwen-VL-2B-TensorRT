cmake_minimum_required(VERSION 3.8)
project(edge_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection.msg" "msg/DetectionFrame.msg" "msg/Snapshot.msg"
  "msg/SceneAnswer.msg" "action/AnalyzeScene.action"
  DEPENDENCIES std_msgs sensor_msgs builtin_interfaces)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
