"""One active action, one latest snapshot, one replaceable pending question."""
import time
import uuid
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String, Empty
from edge_interfaces.msg import Snapshot, SceneAnswer
from edge_interfaces.action import AnalyzeScene
from .ros_utils import sensor_qos, spin_node, stamp_ns, request_from_goal
from .core import Answer, FrameKey, Status, matches_request
from .workstation import (WorkstationMonitor, Observation, EventStore,
                          INSPECTION_QUESTION, parse_roi, person_present)


class Coordinator(Node):
    def __init__(self):
        super().__init__('coordinator')
        defaults = {'auto_ask': True, 'ask_interval_s': 5.0,
                    'question': '请描述画面中主要物体、颜色和相对位置。不确定的内容请说明。',
                    'timeout_s': 15.0, 'max_input_age_s': 3.0,
                    'max_answer_age_s': 20.0, 'max_new_tokens': 128,
                    'inspection_enabled': False, 'occupied_s': 2.0, 'absent_s': 10.0,
                    'max_observation_gap_s': 3.0, 'person_threshold': 0.5,
                    'workstation_roi': '0,0,1,1', 'report_dir': 'runs/workstation'}
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.p = {key: self.get_parameter(key).value for key in defaults}
        if self.p['ask_interval_s'] <= 0:
            raise ValueError('ask_interval_s must be positive')
        self.latest = self.pending = self.active = None
        self.pending_event = self.pending_snapshot = None
        self.monitor = WorkstationMonitor(self.p['occupied_s'], self.p['absent_s'],
            self.p['max_observation_gap_s'], self.p['max_input_age_s'])
        self.roi = parse_roi(self.p['workstation_roi'])
        if not 0 < self.p['person_threshold'] <= 1:
            raise ValueError('person_threshold must be in (0,1]')
        self.store = EventStore(self.p['report_dir']) if self.p['inspection_enabled'] else None
        if self.p['inspection_enabled'] and self.p['auto_ask']:
            raise ValueError('inspection_enabled requires auto_ask=false; use event-triggered analysis')
        self.client = ActionClient(self, AnalyzeScene, '/vlm/analyze_scene')
        self.pub = self.create_publisher(SceneAnswer, '/assistant/answer', 10)
        self.create_subscription(Snapshot, '/perception/snapshot', self.on_snapshot, sensor_qos())
        self.create_subscription(String, '/assistant/question', self.on_question, 10)
        self.create_subscription(Empty, '/assistant/cancel', self.on_cancel, 10)
        self.create_timer(0.1, self.tick)
        self.create_timer(self.p['ask_interval_s'], self.auto_ask)
        # All ROS callbacks use the default mutually exclusive group.

    def on_snapshot(self, msg):
        if (self.latest is not None and msg.session_id == self.latest.session_id
                and msg.frame_number <= self.latest.frame_number):
            return
        self.latest = msg  # Replaces pending snapshot; active goal owns its old snapshot.
        if not self.p['inspection_enabled']:
            return
        try:
            if msg.clock_domain != 'ros':
                raise ValueError('Inspection requires ROS system timestamps')
            present = person_present(msg.detections, msg.original_width, msg.original_height,
                                     self.roi, self.p['person_threshold'])
            event = self.monitor.update(Observation(msg.session_id, msg.frame_number,
                stamp_ns(msg.header.stamp), present), self.get_clock().now().nanoseconds)
            self.invalidate_inspection()
            if event:
                self.store.write(event, 'TRIGGERED', image=bytes(msg.image.data),
                                 question=INSPECTION_QUESTION, roi=list(self.roi))
                if self.active or self.pending is not None:
                    self.store.write(event, 'SKIPPED_BUSY', reason='Another question is active or pending')
                else:
                    self.pending = INSPECTION_QUESTION
                    self.pending_event, self.pending_snapshot = event, msg
                    self.get_logger().info(f'Inspection event={event.event_id} frame={event.frame}')
        except (ValueError, OSError) as exc:
            self.monitor.reset()
            self.invalidate_inspection()
            self.get_logger().error(f'Inspection input/storage error: {exc}')

    def clear_pending(self, reason):
        if self.pending_event:
            try:
                self.store.write(self.pending_event, 'SUPERSEDED', reason=reason)
            except OSError as exc:
                self.get_logger().error(f'Could not persist cancelled event: {exc}')
        self.pending = self.pending_event = self.pending_snapshot = None

    def invalidate_inspection(self):
        fresh = self.monitor.fresh(self.get_clock().now().nanoseconds)
        def invalid(event):
            return event and (not fresh or event.generation != self.monitor.generation)
        if invalid(self.pending_event):
            self.clear_pending('Scene changed, stream restarted, or observations became stale')
        if self.active and invalid(self.active.get('event')):
            self.active['obsolete'] = True
            self.request_cancel()

    def publish_answer(self, msg, active):
        event = active.get('event')
        if event:
            try:
                self.store.write(event, msg.status, request_id=msg.request_id,
                    question=msg.question, answer=msg.answer, model=msg.model,
                    reason=msg.reason, elapsed_ms=msg.elapsed_ms,
                    source_age_ms=msg.source_age_ms)
            except OSError as exc:
                self.get_logger().error(f'Could not persist inspection result: {exc}')
        self.pub.publish(msg)

    def on_question(self, msg):
        if not msg.data.strip() or len(msg.data) > 2000:
            self.get_logger().warning('Question must contain 1..2000 characters')
            return
        self.clear_pending('Replaced by a manual question')
        self.pending = msg.data
        if self.active:
            self.active['obsolete'] = True
            self.request_cancel()

    def on_cancel(self, msg):
        self.clear_pending('Cancelled by user')
        if self.active:
            self.active['obsolete'] = True
            self.request_cancel()

    def auto_ask(self):
        if self.p['auto_ask'] and self.active is None and self.pending is None:
            self.pending = self.p['question']

    def request_cancel(self):
        active = self.active
        if active and active.get('handle') is not None and not active['cancel_sent']:
            active['cancel_sent'] = True
            active['handle'].cancel_goal_async()
            self.get_logger().info('Cancellation requested; waiting for backend termination')

    def diagnostic(self, active, status, reason):
        msg = SceneAnswer()
        goal = active['goal']
        msg.header, msg.request_id = goal.snapshot.header, goal.request_id
        msg.session_id, msg.frame_number = goal.snapshot.session_id, goal.snapshot.frame_number
        msg.question, msg.status, msg.reason = goal.question, status, reason
        msg.elapsed_ms = float((time.monotonic() - active['started']) * 1000)
        msg.source_age_ms = float((self.get_clock().now().nanoseconds - stamp_ns(msg.header.stamp)) / 1e6)
        self.publish_answer(msg, active)

    def tick(self):
        if self.p['inspection_enabled']:
            self.invalidate_inspection()
        if self.active:
            elapsed = time.monotonic() - self.active['started']
            if elapsed > self.p['timeout_s']:
                self.active['expired'] = True
                self.request_cancel()
            if elapsed > self.p['timeout_s'] + 5 and not self.active['unconfirmed_reported']:
                self.active['unconfirmed_reported'] = True
                self.diagnostic(self.active, 'TIMEOUT_UNCONFIRMED',
                                'Waiting for action termination; no new goal will be dispatched')
            return  # Never start a second job because cancellation was merely requested.
        if self.pending is None or self.latest is None or not self.client.server_is_ready():
            return
        snapshot = self.pending_snapshot if self.pending_event else self.latest
        age = (self.get_clock().now().nanoseconds - stamp_ns(snapshot.header.stamp)) / 1e9
        if snapshot.clock_domain != 'ros' or age < -0.05 or age > self.p['max_input_age_s']:
            if self.pending_event:
                self.clear_pending('Event snapshot expired before dispatch')
            return  # Keep the question, wait for a fresh snapshot.
        goal = AnalyzeScene.Goal()
        goal.request_id = str(uuid.uuid4())
        goal.snapshot, goal.question = snapshot, self.pending
        goal.max_new_tokens = self.p['max_new_tokens']
        goal.timeout_s, goal.max_input_age_s = self.p['timeout_s'], self.p['max_input_age_s']
        goal.max_answer_age_s = self.p['max_answer_age_s']
        event = self.pending_event
        self.pending = self.pending_event = self.pending_snapshot = None
        active = dict(goal=goal, started=time.monotonic(), handle=None,
                      obsolete=False, expired=False, cancel_sent=False, unconfirmed_reported=False,
                      event=event)
        self.active = active
        try:
            future = self.client.send_goal_async(goal, feedback_callback=self.on_feedback)
            future.add_done_callback(lambda result: self.on_accepted(result, goal.request_id))
        except Exception as exc:
            self.diagnostic(active, 'ERROR', str(exc))
            self.active = None

    def on_feedback(self, feedback):
        self.get_logger().debug(f'VLM {feedback.feedback.phase}: {feedback.feedback.elapsed_ms:.0f} ms')

    def on_accepted(self, future, request_id):
        if not self.active or self.active['goal'].request_id != request_id:
            return
        try:
            handle = future.result()
            if not handle.accepted:
                self.diagnostic(self.active, 'REJECTED', 'VLM busy or invalid/stale input; inspect server log')
                self.active = None
                return
            self.active['handle'] = handle
            result_future = handle.get_result_async()
            result_future.add_done_callback(lambda result: self.on_result(result, request_id))
            if self.active['obsolete'] or self.active['expired']:
                self.request_cancel()
        except Exception as exc:
            self.diagnostic(self.active, 'ERROR', str(exc))
            self.active = None

    def on_result(self, future, request_id):
        if not self.active or self.active['goal'].request_id != request_id:
            return
        if self.p['inspection_enabled']:
            self.invalidate_inspection()  # also check freshness at result arrival, not only on timer ticks
        active = self.active
        try:
            response = future.result()
            msg = response.result.answer
            goal = active['goal']
            core_answer = Answer(msg.request_id, FrameKey(msg.session_id, msg.frame_number),
                                 stamp_ns(msg.header.stamp), Status(msg.status), msg.answer,
                                 msg.model, msg.elapsed_ms / 1000, msg.reason)
            if not matches_request(core_answer, request_from_goal(goal)):
                self.diagnostic(active, 'ERROR', 'Response/image identity mismatch')
                return
            age = (self.get_clock().now().nanoseconds - stamp_ns(msg.header.stamp)) / 1e9
            if msg.status == 'OK':
                if active['expired']:
                    msg.status, msg.reason = 'TIMEOUT', 'Client deadline expired'
                elif active['obsolete']:
                    msg.status, msg.reason = 'SUPERSEDED', 'Question cancelled/replaced, scene changed, or observation stream stale'
                elif response.status != GoalStatus.STATUS_SUCCEEDED:
                    msg.status, msg.reason = 'ERROR', 'Action terminal state is not SUCCEEDED'
                elif age < -0.05 or age > self.p['max_answer_age_s']:
                    msg.status, msg.reason = 'STALE', 'Snapshot is no longer fresh'
            if msg.status != 'OK':
                msg.answer = ''
            msg.source_age_ms = float(age * 1000)
            self.publish_answer(msg, active)
            self.get_logger().info(f'frame={msg.frame_number} {msg.status}: {msg.answer or msg.reason}')
        except Exception as exc:
            self.diagnostic(active, 'ERROR', str(exc))
        finally:
            self.active = None

    def request_stop(self):
        self.clear_pending('Coordinator shutting down')
        if self.active and self.active.get('event'):
            try:
                self.store.write(self.active['event'], 'INTERRUPTED', reason='Coordinator shutting down')
            except OSError as exc:
                self.get_logger().error(f'Could not persist interrupted event: {exc}')
        # Backend process also receives SIGINT from launch and stops cooperatively.


def main():
    spin_node(Coordinator)
