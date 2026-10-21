std_msgs/Header header
string session_id
uint64 frame_number
string clock_domain
string timestamp_source
sensor_msgs/CompressedImage image
# Metadata from the SAME source frame. The VLM receives the image, not these boxes.
uint32 original_width
uint32 original_height
edge_interfaces/Detection[] detections
