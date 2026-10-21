"""Exercise the actual coordinator callbacks with ROS transport doubles.

These tests verify application logic, NOT DDS or a running ROS Action server.
"""
import importlib
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/edge_assistant'))
from edge_assistant.core import FrameKey, Snapshot as CoreSnapshot, Request
from edge_assistant.workstation import WorkstationMonitor, EventStore


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


def request(goal):
    s = goal.snapshot
    return Request(goal.request_id, goal.question,
        CoreSnapshot(FrameKey(s.session_id, s.frame_number), s.header.stamp, bytes(s.image.data)))


class CoordinatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        doubles = {
            'action_msgs.msg': module('action_msgs.msg', GoalStatus=NS(STATUS_SUCCEEDED=4)),
            'rclpy.action': module('rclpy.action', ActionClient=MagicMock),
            'rclpy.node': module('rclpy.node', Node=object),
            'std_msgs.msg': module('std_msgs.msg', String=NS, Empty=NS),
            'edge_interfaces.msg': module('edge_interfaces.msg', Snapshot=NS,
                SceneAnswer=lambda: NS(answer='', model='', elapsed_ms=0., source_age_ms=0.)),
            'edge_interfaces.action': module('edge_interfaces.action', AnalyzeScene=NS(Goal=NS)),
            'edge_assistant.ros_utils': module('edge_assistant.ros_utils', sensor_qos=lambda: None,
                spin_node=lambda _: None, stamp_ns=lambda x: x, request_from_goal=request),
        }
        with patch.dict(sys.modules, doubles):
            sys.modules.pop('edge_assistant.coordinator', None)
            cls.Coordinator = importlib.import_module('edge_assistant.coordinator').Coordinator

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.node = self.Coordinator.__new__(self.Coordinator)
        n = self.node
        n.p = dict(inspection_enabled=True, person_threshold=.5, max_input_age_s=3.,
                   timeout_s=60., max_answer_age_s=90., max_new_tokens=128, auto_ask=False)
        n.latest = n.pending = n.active = n.pending_event = n.pending_snapshot = None
        n.monitor = WorkstationMonitor(1, 2, 3, 3)
        n.roi = (0,0,1,1)
        n.store = EventStore(self.tmp.name)
        n.client = MagicMock()
        n.client.server_is_ready.return_value = True
        n.pub = MagicMock()
        n.get_logger = lambda: MagicMock()
        self.now = 1_000_000_000_000
        n.get_clock = lambda: NS(now=lambda: NS(nanoseconds=self.now))

    def frame(self, second, present):
        self.now = 1_000_000_000_000+second*1_000_000_000
        detections = [NS(label='person', confidence=.9, xmin=10,ymin=10,xmax=60,ymax=80)] if present else []
        msg = NS(session_id='session', frame_number=second, header=NS(stamp=self.now),
                 clock_domain='ros', original_width=100, original_height=100,
                 detections=detections, image=NS(data=b'fake-jpeg'))
        self.node.on_snapshot(msg)
        return msg

    def trigger(self):
        self.frame(0, True); self.frame(1, True)
        self.frame(2, False); self.frame(3, False)
        return self.frame(4, False)

    def test_event_dispatch_pins_exact_snapshot(self):
        trigger = self.trigger()
        self.frame(5, False)
        self.node.tick()
        self.assertIs(self.node.active['goal'].snapshot, trigger)
        self.assertEqual(self.node.active['event'].frame, 4)
        self.node.client.send_goal_async.assert_called_once()

    def test_return_clears_pending_event(self):
        self.trigger(); self.frame(5, True); self.node.tick()
        self.assertIsNone(self.node.pending_event)
        self.node.client.send_goal_async.assert_not_called()

    def test_return_cancels_inflight_but_does_not_free_slot(self):
        self.trigger(); self.node.tick()
        handle = MagicMock()
        self.node.active['handle'] = handle
        self.frame(5, True)
        self.assertTrue(self.node.active['obsolete'])
        self.assertIsNotNone(self.node.active)
        handle.cancel_goal_async.assert_called_once()

    def test_manual_question_replaces_event_snapshot(self):
        self.trigger()
        self.node.on_question(NS(data='What is on the desk?'))
        latest = self.frame(5, False)
        self.node.tick()
        self.assertIsNone(self.node.active['event'])
        self.assertIs(self.node.active['goal'].snapshot, latest)

    def test_busy_event_is_recorded_without_overwriting_question(self):
        self.node.pending = 'manual question'
        self.trigger()
        self.assertEqual(self.node.pending, 'manual question')
        self.assertIsNone(self.node.pending_event)
        self.assertIn('SKIPPED_BUSY', Path(self.tmp.name,'events.jsonl').read_text(encoding='utf-8'))

    def test_stale_event_is_not_silently_rebound(self):
        self.trigger()
        self.node.client.server_is_ready.return_value = False
        for second in range(5, 9): self.frame(second, False)
        self.node.client.server_is_ready.return_value = True
        self.node.tick()
        self.assertIsNone(self.node.pending)
        self.node.client.send_goal_async.assert_not_called()

    def test_result_freshness_checked_even_without_timer_tick(self):
        self.trigger(); self.node.tick()
        goal = self.node.active['goal']
        self.now += 4_000_000_000
        answer = NS(request_id=goal.request_id, session_id='session', frame_number=4,
                    header=goal.snapshot.header, question=goal.question, status='OK',
                    answer='obsolete report', model='test', elapsed_ms=4000., reason='')
        future = MagicMock()
        future.result.return_value = NS(status=4, result=NS(answer=answer))
        self.node.on_result(future, goal.request_id)
        self.assertEqual(answer.status, 'SUPERSEDED')
        self.assertEqual(answer.answer, '')
        self.assertIsNone(self.node.active)


if __name__ == '__main__': unittest.main()
