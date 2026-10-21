import dataclasses
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/edge_assistant'))
from edge_assistant.workstation import (WorkstationMonitor, Observation, EventStore,
                                       parse_roi, person_present)
from edge_assistant.rtdetr_backend import decode_rows

BASE = 1_000_000_000_000


class WorkstationTests(unittest.TestCase):
    def setUp(self):
        self.monitor = WorkstationMonitor(2, 3, 2, 2)

    def feed(self, second, present, session='camera', frame=None, delay=0):
        stamp = BASE+int(second*1e9)
        return self.monitor.update(Observation(session, second if frame is None else frame,
                                               stamp, present), stamp+int(delay*1e9))

    def occupied(self):
        for second in range(3):
            self.assertIsNone(self.feed(second, True))
        self.assertEqual(self.monitor.state, 'OCCUPIED')

    def event(self):
        self.occupied()
        for second in range(3, 6):
            self.assertIsNone(self.feed(second, False))
        return self.feed(6, False)

    def test_initial_empty_never_triggers(self):
        for second in range(20):
            self.assertIsNone(self.feed(second, False))

    def test_one_event_per_occupied_absent_cycle(self):
        event = self.event()
        self.assertEqual(event.frame, 6)
        self.assertEqual(event.absence_started_ns, BASE+3_000_000_000)
        for second in range(7, 20):
            self.assertIsNone(self.feed(second, False))

    def test_short_person_detection_does_not_arm(self):
        self.feed(0, True)
        for second in range(1, 10):
            self.assertIsNone(self.feed(second, False))

    def test_brief_occlusion_does_not_trigger(self):
        self.occupied()
        self.feed(3, False); self.feed(4, False); self.feed(5, True)
        self.assertEqual(self.monitor.state, 'OCCUPIED')
        self.feed(6, False); self.feed(7, False); self.feed(8, False)
        self.assertIsNotNone(self.feed(9, False))

    def test_return_invalidates_old_event_and_rearms(self):
        event = self.event()
        self.feed(7, True)
        self.assertNotEqual(event.generation, self.monitor.generation)
        self.feed(8, True); self.feed(9, True)
        self.feed(10, False); self.feed(11, False); self.feed(12, False)
        self.assertIsNotNone(self.feed(13, False))

    def test_stream_gap_not_counted_as_absence(self):
        self.occupied(); self.feed(3, False)
        self.assertIsNone(self.feed(10, False))
        self.assertEqual(self.monitor.state, 'EMPTY')

    def test_stream_restart_requires_new_occupancy(self):
        self.occupied()
        self.assertIsNone(self.feed(3, False, session='restart', frame=0))
        for second in range(4, 10):
            self.assertIsNone(self.feed(second, False, session='restart'))

    def test_stale_and_future_inputs_reset(self):
        for delay in (3, -.1):
            self.setUp(); self.occupied()
            self.assertIsNone(self.feed(3, False, delay=delay))
            self.assertEqual(self.monitor.state, 'EMPTY')

    def test_duplicate_and_out_of_order_do_not_advance(self):
        self.occupied(); self.feed(3, False)
        self.assertIsNone(self.feed(6, False, frame=3))
        self.assertEqual(self.monitor.last.frame, 3)
        self.assertIsNone(self.feed(2, True))

    def test_silence_is_stale_not_an_event(self):
        self.occupied()
        self.assertFalse(self.monitor.fresh(BASE+10_000_000_000))
        self.assertEqual(self.monitor.state, 'OCCUPIED')

    def test_invalid_parameters(self):
        for value in (0, -1, float('nan')):
            with self.assertRaises(ValueError): WorkstationMonitor(absent_s=value)
        for roi in ('0,0,0,1', '0,0,2,1', '0,0,nan,1', '0,1'):
            with self.assertRaises(ValueError): parse_roi(roi)

    def test_person_roi_score_and_teaching_rejection(self):
        person = SimpleNamespace(label='person', confidence=.9, xmin=0, ymin=0, xmax=20, ymax=20)
        roi = parse_roi('0,0,.5,1')
        self.assertTrue(person_present([person], 100, 100, roi))
        self.assertFalse(person_present([person], 100, 100, parse_roi('.5,0,1,1')))
        person.confidence=.1
        self.assertFalse(person_present([person], 100, 100, roi))
        person.label='red_region_teaching_detector'
        with self.assertRaises(ValueError): person_present([person], 100, 100, roi)

    def test_store_retains_image_and_terminal_status(self):
        event = self.event()
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(directory)
            store.write(event, 'TRIGGERED', image=b'jpeg-test-only', question='工位检查', roi=[0,0,1,1])
            store.write(event, 'TIMEOUT', answer='', reason='test deadline')
            record = json.loads((Path(directory)/(event.event_id+'.json')).read_text(encoding='utf-8'))
            self.assertEqual(record['frame'], event.frame)
            self.assertEqual(record['status'], 'TIMEOUT')
            self.assertTrue(record['image'].endswith('.jpg'))
            self.assertEqual(record['question'], '工位检查')
            self.assertEqual(record['roi'], [0,0,1,1])
            self.assertEqual(len((Path(directory)/'events.jsonl').read_text(encoding='utf-8').splitlines()), 2)
            with self.assertRaises(ValueError):
                store.write(dataclasses.replace(event, event_id='../escape'), 'OK')


class DetrContractTests(unittest.TestCase):
    def test_normalized_xyxy_not_letterbox(self):
        boxes = decode_rows([[.1,.2,.8,.9,.95,24]], 1000, 500)
        self.assertEqual(boxes[0]['xyxy'], [100,100,800,450])
        self.assertEqual(boxes[0]['class_id'], 24)

    def test_no_nms_and_threshold(self):
        row = [0,0,1,1,.9,0]
        self.assertEqual(len(decode_rows([row,row,[0,0,1,1,.1,0]], 100, 50)), 2)

    def test_invalid_export_rejected(self):
        for row in ([0]*84, [0,0,1,1,.9,80], [0,0,1,1,.9,.2], [0,0,1,1,float('nan'),0]):
            with self.assertRaises(ValueError): decode_rows([row], 100, 50)

    def test_clipping_and_empty_box(self):
        boxes = decode_rows([[-.1,-.2,1.1,1.2,.9,0],[1,1,0,0,.9,0]], 100, 50)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]['xyxy'], [0,0,100,50])


if __name__ == '__main__':
    unittest.main()
