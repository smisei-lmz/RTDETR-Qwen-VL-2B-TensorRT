import threading
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from edge_interfaces.action import AnalyzeScene
from .backends import make_backend
from .core import SingleFlight, Status, validate_request, run_request
from .ros_utils import request_from_goal, answer_message, spin_node


class VlmServer(Node):
    def __init__(self):
        super().__init__('vlm_server')
        params = {'backend': 'mock', 'model': 'Qwen/Qwen2-VL-2B-Instruct',
                  'device': 'cuda:0', 'mock_delay_s': 2.0}
        for key, default in params.items():
            self.declare_parameter(key, default)
        self.stopping = threading.Event()
        self.flight = SingleFlight()
        self.get_logger().info('Loading backend BEFORE accepting action requests...')
        self.backend = make_backend(*(self.get_parameter(key).value for key in params))
        self.server = ActionServer(
            self, AnalyzeScene, '/vlm/analyze_scene', execute_callback=self.execute,
            goal_callback=self.accept, cancel_callback=self.cancel,
            callback_group=ReentrantCallbackGroup(), result_timeout=30)
        self.get_logger().info(f'VLM action ready: {self.backend.name}; concurrency=1')

    def accept(self, goal):
        try:
            validate_request(request_from_goal(goal), self.get_clock().now().nanoseconds)
        except (ValueError, TypeError) as exc:
            self.get_logger().warning(f'Rejected invalid/stale request: {exc}')
            return GoalResponse.REJECT
        if self.stopping.is_set() or not self.flight.reserve():
            self.get_logger().warning('Rejected request: backend busy or stopping')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel(self, goal_handle):
        return CancelResponse.ACCEPT

    def execute(self, goal_handle):
        goal = goal_handle.request
        try:
            req = request_from_goal(goal)

            def feedback(phase, elapsed):
                if not self.stopping.is_set() and rclpy.ok():
                    msg = AnalyzeScene.Feedback()
                    msg.phase, msg.elapsed_ms = phase, float(elapsed * 1000)
                    goal_handle.publish_feedback(msg)

            answer = run_request(req, self.backend,
                                 lambda: goal_handle.is_cancel_requested or self.stopping.is_set(),
                                 lambda: self.get_clock().now().nanoseconds, feedback)
            result = AnalyzeScene.Result()
            result.answer = answer_message(answer, goal, self.get_clock().now().nanoseconds)
            if goal_handle.is_cancel_requested:
                # Close the small race where cancellation arrives after generation returns.
                result.answer.status, result.answer.answer = Status.CANCELLED.value, ''
                goal_handle.canceled()
            elif answer.status == Status.OK:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            self.get_logger().info(
                f'{req.request_id} frame={req.snapshot.key.frame_id} {result.answer.status}')
            return result
        finally:
            # Never free the single model lane merely because a timeout was requested.
            self.flight.release()

    def request_stop(self):
        self.stopping.set()


def main():
    spin_node(VlmServer, threads=3)
