import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from edge_interfaces.msg import SceneAnswer
from .core import FrameKey, Snapshot, Request


def sensor_qos():
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                      reliability=ReliabilityPolicy.BEST_EFFORT)


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def request_from_goal(goal):
    msg = goal.snapshot
    mime = 'image/png' if msg.image.format == 'png' else 'image/jpeg'
    if msg.image.format not in ('jpeg', 'png'):
        raise ValueError('only jpeg/png snapshots supported')
    snapshot = Snapshot(FrameKey(msg.session_id, msg.frame_number), stamp_ns(msg.header.stamp),
                        bytes(msg.image.data), mime, msg.clock_domain, msg.timestamp_source)
    return Request(goal.request_id, goal.question, snapshot, goal.max_new_tokens,
                   goal.timeout_s, goal.max_input_age_s, goal.max_answer_age_s)


def answer_message(answer, goal, now_ns):
    msg = SceneAnswer()
    msg.header = goal.snapshot.header
    msg.request_id = answer.request_id
    msg.session_id = answer.key.session_id
    msg.frame_number = answer.key.frame_id
    msg.question = goal.question
    msg.status = answer.status.value
    msg.answer, msg.model, msg.reason = answer.text, answer.model, answer.reason
    msg.elapsed_ms = float(answer.elapsed_s * 1000)
    msg.source_age_ms = float((now_ns - answer.captured_ns) / 1e6)
    return msg


def spin_node(factory, threads=3):
    rclpy.init()
    node = None
    executor = MultiThreadedExecutor(num_threads=threads)
    try:
        node = factory()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node is not None and hasattr(node, 'request_stop'):
            node.request_stop()
        # Cooperative model cancellation cannot preempt an already running GPU kernel.
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
