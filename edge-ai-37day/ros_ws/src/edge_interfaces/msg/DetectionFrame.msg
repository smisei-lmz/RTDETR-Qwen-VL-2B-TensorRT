# stamp = host acquisition ROS time; frame_id = optical coordinate frame name.
std_msgs/Header header
string session_id
uint64 frame_number
uint32 image_width
uint32 image_height
edge_interfaces/Detection[] detections
string detector
float32 processing_ms
