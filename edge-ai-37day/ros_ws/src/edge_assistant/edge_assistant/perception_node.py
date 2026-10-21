"""Owned CPU frames for the first PC functional demo; not a DMA-BUF backend."""
import threading
import time
import uuid
import cv2
import numpy as np
from rclpy.node import Node
from edge_interfaces.msg import Detection, DetectionFrame, Snapshot
from .core import LatestSlot
from .ros_utils import sensor_qos, spin_node
from .rtdetr_backend import RTDetrBackend, MODEL_ID


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception')
        defaults = {'source': 'synthetic', 'fps': 30.0, 'snapshot_hz': 1.0,
                    'snapshot_width': 640, 'detector': 'color', 'weights': MODEL_ID, 'device': '0'}
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.params = {key: self.get_parameter(key).value for key in defaults}
        if self.params['fps'] <= 0 or self.params['snapshot_hz'] <= 0:
            raise ValueError('fps and snapshot_hz must be positive')
        if not 64 <= self.params['snapshot_width'] <= 1280:
            raise ValueError('snapshot_width must be 64..1280')
        self.session = str(uuid.uuid4())
        self.latest = LatestSlot()
        self.stopping = threading.Event()
        self.capture_error = None
        self.captured = self.processed = self.snapshots = 0
        self.last_snapshot = 0.0
        self.model = None
        if self.params['detector'] == 'rtdetr':
            self.model = RTDetrBackend(self.params['weights'], self.params['device'])
        elif self.params['detector'] != 'color':
            raise ValueError('detector must be color or rtdetr')
        self.detection_pub = self.create_publisher(DetectionFrame, '/perception/detections', sensor_qos())
        self.snapshot_pub = self.create_publisher(Snapshot, '/perception/snapshot', sensor_qos())
        self.thread = threading.Thread(target=self.capture_loop, name='capture', daemon=True)
        self.thread.start()
        # Default mutually exclusive callback group: one detector invocation at a time.
        self.create_timer(0.005, self.process_latest)
        self.create_timer(2.0, self.report)

    def capture_loop(self):
        source = self.params['source']
        cap = None
        try:
            if source != 'synthetic':
                cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
                if not cap.isOpened():
                    raise RuntimeError(f'cannot open source: {source}')
            period = 1.0 / self.params['fps']
            while not self.stopping.is_set():
                started = time.monotonic()
                if cap is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    left = 20 + (self.captured * 5) % 420
                    cv2.rectangle(frame, (left, 150), (left + 120, 270), (0, 0, 255), -1)
                    cv2.putText(frame, 'Synthetic red square', (15, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                else:
                    ok, frame = cap.read()
                    if not ok:
                        if source.isdigit():
                            raise RuntimeError('camera read failed')
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError('video cannot be rewound/read')
                captured_stamp = self.get_clock().now().to_msg()
                self.captured += 1
                # Independent owned ndarray. Capture never mutates it after publication.
                self.latest.put((self.captured, captured_stamp, frame))
                self.stopping.wait(max(0.0, period - (time.monotonic() - started)))
        except Exception as exc:
            self.capture_error = str(exc)
            self.stopping.set()
        finally:
            if cap is not None:
                cap.release()

    def detect(self, frame):
        detections = []
        if self.model is not None:
            output = self.model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Library-managed GPU work completes before .cpu() results are consumed.
            for row in output:
                x1, y1, x2, y2 = row['xyxy']
                detections.append(Detection(label=self.model.names[row['class_id']],
                    confidence=float(row['score']), xmin=float(x1), ymin=float(y1),
                    xmax=float(x2), ymax=float(y2)))
            return detections
        # A real pixel-based teaching detector, not a trained object detector.
        b, g, r = cv2.split(frame)
        mask = ((r > 160) & (g < 100) & (b < 100)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= 100:
                detections.append(Detection(label='red_region_teaching_detector', confidence=1.0,
                    xmin=float(x), ymin=float(y), xmax=float(x + w), ymax=float(y + h)))
        return detections

    def process_latest(self):
        value = self.latest.take()
        if value is None:
            return
        frame_number, stamp, frame = value
        start = time.monotonic()
        try:
            detections = self.detect(frame)
        except Exception as exc:
            self.get_logger().error(f'Detection failed for frame {frame_number}: {exc}')
            return
        result = DetectionFrame()
        result.header.stamp, result.header.frame_id = stamp, 'camera_optical_frame'
        result.session_id, result.frame_number = self.session, frame_number
        result.image_height, result.image_width = frame.shape[:2]
        result.detections, result.detector = detections, self.params['detector']
        result.processing_ms = float((time.monotonic() - start) * 1000)
        self.detection_pub.publish(result)
        self.processed += 1
        if start - self.last_snapshot < 1.0 / self.params['snapshot_hz']:
            return
        scale = min(1.0, self.params['snapshot_width'] / frame.shape[1], 720 / frame.shape[0])
        thumbnail = cv2.resize(frame, (max(1, round(frame.shape[1] * scale)),
                                       max(1, round(frame.shape[0] * scale))))
        ok, encoded = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok or encoded.size > 1024 * 1024:
            self.get_logger().warning('Snapshot encoding failed or image > 1 MiB')
            return
        snapshot = Snapshot()
        snapshot.header = result.header
        snapshot.session_id, snapshot.frame_number = self.session, frame_number
        snapshot.clock_domain, snapshot.timestamp_source = 'ros', 'host_acquire'
        snapshot.image.header, snapshot.image.format = result.header, 'jpeg'
        snapshot.image.data = encoded.tobytes()
        snapshot.original_width, snapshot.original_height = result.image_width, result.image_height
        snapshot.detections = detections
        self.snapshot_pub.publish(snapshot)
        self.last_snapshot = start
        self.snapshots += 1

    def report(self):
        if self.capture_error:
            self.get_logger().error(self.capture_error)
        self.get_logger().info(
            f'captured={self.captured} processed={self.processed} '
            f'replaced_pending={self.latest.replaced} snapshots={self.snapshots}')

    def request_stop(self):
        self.stopping.set()
        self.thread.join(timeout=3.0)
        if self.thread.is_alive():
            self.get_logger().error('Capture read did not exit; driver/process recovery may be required')


def main():
    spin_node(PerceptionNode)
