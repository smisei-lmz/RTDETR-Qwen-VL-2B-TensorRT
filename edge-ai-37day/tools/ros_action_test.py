"""Integration test against a running mock VLM server (2 s delay, auto_ask=false)."""
import copy
import json
import time
import uuid
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile,ReliabilityPolicy
from edge_interfaces.action import AnalyzeScene
from edge_interfaces.msg import Snapshot


def main():
    rclpy.init();node=Node('course_action_test');latest=[None]
    sub=node.create_subscription(Snapshot,'/perception/snapshot',lambda m:latest.__setitem__(0,m),
                                 QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))
    client=ActionClient(node,AnalyzeScene,'/vlm/analyze_scene')
    def wait(future,timeout=15):
        end=time.monotonic()+timeout
        while not future.done() and time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.05)
        if not future.done():raise RuntimeError('action operation timed out')
        return future.result()
    def goal(timeout=8):
        msg=AnalyzeScene.Goal();msg.request_id=str(uuid.uuid4());msg.snapshot=copy.deepcopy(latest[0])
        msg.question='Integration test: describe this image';msg.max_new_tokens=32;msg.timeout_s=float(timeout)
        msg.max_input_age_s=3.;msg.max_answer_age_s=15.;return msg
    try:
        if not client.wait_for_server(timeout_sec=15):raise RuntimeError('no VLM action server')
        end=time.monotonic()+10
        while latest[0] is None and time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.1)
        if latest[0] is None:raise RuntimeError('no fresh snapshot')
        first=wait(client.send_goal_async(goal()));assert first.accepted
        busy=wait(client.send_goal_async(goal()));assert not busy.accepted,'concurrent goal was accepted'
        cancel=wait(first.cancel_goal_async());assert cancel.goals_canceling,'cancel was rejected'
        cancelled=wait(first.get_result_async()).result.answer;assert cancelled.status=='CANCELLED'
        timed=wait(client.send_goal_async(goal(.1)));assert timed.accepted
        timed_result=wait(timed.get_result_async()).result.answer;assert timed_result.status=='TIMEOUT'
        last_goal=goal();last=wait(client.send_goal_async(last_goal));assert last.accepted
        answer=wait(last.get_result_async()).result.answer
        assert answer.status=='OK' and answer.request_id==last_goal.request_id
        assert answer.session_id==last_goal.snapshot.session_id and answer.frame_number==last_goal.snapshot.frame_number
        assert 'MOCK' in answer.answer,'this test is for the mock server only'
        print(json.dumps(dict(busy_rejected=True,cancelled=True,timeout=True,identity_matched=True)))
    finally:node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
