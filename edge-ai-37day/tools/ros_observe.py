import argparse
import json
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy
from edge_interfaces.msg import DetectionFrame,Snapshot,SceneAnswer


def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=8)
    p.add_argument('--snapshots',action='store_true');p.add_argument('--answers',action='store_true');args=p.parse_args()
    rclpy.init();node=Node('course_observer');qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
    counts={'detections':0,'snapshots':0,'answers':0};errors=[];identities=[]
    def detection(msg):
        counts['detections']+=1
        if not msg.session_id or msg.image_width==0:errors.append('invalid detection metadata')
    def snapshot(msg):
        counts['snapshots']+=1
        if msg.header.stamp!=msg.image.header.stamp or not msg.image.data:errors.append('snapshot/header mismatch')
    def answer(msg):
        if msg.status=='OK':
            counts['answers']+=1;identities.append([msg.request_id,msg.session_id,msg.frame_number])
            if not msg.request_id or not msg.session_id or not msg.answer:errors.append('invalid answer identity/text')
    subscriptions=[node.create_subscription(DetectionFrame,'/perception/detections',detection,qos),
                   node.create_subscription(Snapshot,'/perception/snapshot',snapshot,qos),
                   node.create_subscription(SceneAnswer,'/assistant/answer',answer,10)]
    end=time.monotonic()+args.seconds
    try:
        while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.1)
        if not counts['detections']:errors.append('no detections messages')
        if args.snapshots and not counts['snapshots']:errors.append('no snapshots')
        if args.answers and not counts['answers']:errors.append('no successful answers')
        print(json.dumps(dict(counts=counts,errors=errors,answer_identities=identities)))
        if errors:raise RuntimeError('; '.join(errors))
    finally:node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
