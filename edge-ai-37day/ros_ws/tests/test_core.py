import dataclasses
from pathlib import Path
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'edge_assistant'))
from edge_assistant.core import (FrameKey, Snapshot, Request, Answer, Status,
                                 LatestSlot, SingleFlight, StopControl,
                                 validate_request, run_request, matches_request)
from edge_assistant.backends import MockBackend


def make_request(**kwargs):
    return Request('request-1', 'Describe this frame',
                   Snapshot(FrameKey('camera-session-a', 7), time.time_ns(), b'image'), **kwargs)


class CoreTests(unittest.TestCase):
    def test_valid_request(self):
        validate_request(make_request(), time.time_ns())

    def test_stale_input_rejected(self):
        req = make_request(max_input_age_s=1)
        with self.assertRaises(ValueError):
            validate_request(req, req.snapshot.captured_ns + 2_000_000_000)

    def test_future_timestamp_rejected(self):
        req = make_request()
        with self.assertRaises(ValueError):
            validate_request(req, req.snapshot.captured_ns - 1_000_000_000)

    def test_payload_and_generation_limits(self):
        req = make_request()
        bad = [dataclasses.replace(req, max_new_tokens=0),
               dataclasses.replace(req, timeout_s=float('nan')),
               dataclasses.replace(req, question=''),
               dataclasses.replace(req, snapshot=dataclasses.replace(req.snapshot, image=b'x' * (1024*1024+1))),
               dataclasses.replace(req, snapshot=dataclasses.replace(req.snapshot, clock_domain='sensor-private'))]
        for request in bad:
            with self.subTest(request=request.request_id), self.assertRaises(ValueError):
                validate_request(request, time.time_ns())

    def test_pending_overwrite_does_not_mutate_active(self):
        slot = LatestSlot()
        req = make_request()
        slot.put(req.snapshot)
        active = slot.take()
        for frame in range(8, 1000):
            slot.put(dataclasses.replace(req.snapshot, key=FrameKey('camera-session-a', frame)))
        self.assertEqual(active.key.frame_id, 7)
        self.assertEqual(slot.take().key.frame_id, 999)
        self.assertIsNone(slot.take())
        self.assertEqual(slot.replaced, 991)

    def test_single_flight_concurrent_admission(self):
        gate, barrier = SingleFlight(), threading.Barrier(12)
        results = []
        lock = threading.Lock()
        def reserve():
            barrier.wait()
            result = gate.reserve()
            with lock:
                results.append(result)
        threads = [threading.Thread(target=reserve) for _ in range(12)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertEqual(sum(results), 1)
        gate.release()
        self.assertTrue(gate.reserve())
        gate.release()
        with self.assertRaises(RuntimeError): gate.release()

    def test_monotonic_deadline(self):
        now = [10.0]
        stop = StopControl(2, lambda: False, lambda: now[0])
        now[0] = 11.9
        self.assertIsNone(stop.reason())
        now[0] = 12.0
        self.assertEqual(stop.reason(), Status.TIMEOUT)

    def test_cancel_before_generation(self):
        req = make_request()
        answer = run_request(req, MockBackend(5), lambda: True, time.time_ns)
        self.assertEqual(answer.status, Status.CANCELLED)
        self.assertEqual(answer.text, '')
        self.assertTrue(matches_request(answer, req))

    def test_cancel_during_generation(self):
        event = threading.Event()
        timer = threading.Timer(0.03, event.set)
        timer.start()
        try:
            answer = run_request(make_request(), MockBackend(5), event.is_set, time.time_ns)
        finally:
            timer.join()
        self.assertEqual(answer.status, Status.CANCELLED)
        self.assertLess(answer.elapsed_s, 2)

    def test_timeout_drops_partial_answer(self):
        answer = run_request(make_request(timeout_s=0.03), MockBackend(5), lambda: False, time.time_ns)
        self.assertEqual(answer.status, Status.TIMEOUT)
        self.assertEqual(answer.text, '')

    def test_response_stale_after_generation(self):
        req = make_request(max_answer_age_s=1)
        now = [req.snapshot.captured_ns]
        class SlowScene:
            name = 'test'
            def generate(self, request, stop, feedback):
                now[0] += 2_000_000_000
                return 'outdated observation'
        answer = run_request(req, SlowScene(), lambda: False, lambda: now[0])
        self.assertEqual(answer.status, Status.STALE)
        self.assertEqual(answer.text, '')

    def test_backend_error_does_not_produce_success(self):
        class Failed:
            name = 'test'
            def generate(self, *args):
                raise RuntimeError('injected OOM')
        answer = run_request(make_request(), Failed(), lambda: False, time.time_ns)
        self.assertEqual(answer.status, Status.ERROR)
        self.assertIn('injected OOM', answer.reason)

    def test_result_identity_rejects_restart_and_wrong_request(self):
        req = make_request()
        answer = run_request(req, MockBackend(0), lambda: False, time.time_ns)
        self.assertTrue(matches_request(answer, req))
        self.assertFalse(matches_request(dataclasses.replace(answer, request_id='different'), req))
        self.assertFalse(matches_request(dataclasses.replace(answer, key=FrameKey('camera-session-b', 7)), req))
        self.assertFalse(matches_request(dataclasses.replace(answer, captured_ns=0), req))
        self.assertIn('MOCK', answer.text)


if __name__ == '__main__':
    unittest.main()
