"""Two real ROS2 nodes exchange five String messages; no camera/model needed."""
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String


def main():
    rclpy.init();talker=Node('lesson_talker');listener=Node('lesson_listener')
    pub=talker.create_publisher(String,'/lesson/hello',10);received=[]
    sub=listener.create_subscription(String,'/lesson/hello',lambda m:received.append(m.data),10)
    timer=talker.create_timer(.2,lambda:pub.publish(String(data='hello from a ROS2 node')))
    executor=SingleThreadedExecutor();executor.add_node(talker);executor.add_node(listener)
    deadline=time.monotonic()+10
    try:
        while len(received)<5 and time.monotonic()<deadline:executor.spin_once(timeout_sec=.1)
        if len(received)<5:raise RuntimeError('DDS did not deliver five messages')
        print('PASS: two ROS2 nodes exchanged five messages')
    finally:
        executor.shutdown();talker.destroy_node();listener.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
