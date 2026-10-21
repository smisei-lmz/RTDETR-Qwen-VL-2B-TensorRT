"""Bridge the ACTUAL C++ pipeline's JSONL results/snapshots into ROS2."""
import base64
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import cv2
import numpy as np
from rclpy.node import Node
from edge_interfaces.msg import Detection, DetectionFrame, Snapshot
from .core import LatestSlot
from .ros_utils import sensor_qos, spin_node


class CppBridge(Node):
    def __init__(self):
        super().__init__('cpp_perception_bridge')
        defaults = {'pipeline_exe': '', 'producer_kind': 'pc', 'backend': 'cpu',
                    'engine': '', 'source': 'synthetic', 'camera': '/dev/video0',
                    'camera_format': 'YUYV', 'camera_mode': 'shared', 'matrix': 601,
                    'camera_width':640,'camera_height':480,'full_range':False,
                    'names_json': '', 'snapshot_hz': 1.0}
        for k,v in defaults.items(): self.declare_parameter(k,v)
        p={k:self.get_parameter(k).value for k in defaults}
        if self.has_parameter('use_sim_time') and self.get_parameter('use_sim_time').value:
            raise RuntimeError('C++ source stamps use system time; this bridge does not support use_sim_time')
        executable=Path(p['pipeline_exe']).expanduser()
        if not executable.is_file(): raise RuntimeError('pipeline_exe must point to the built C++ executable')
        command=[str(executable.resolve()),'--backend',p['backend'],'--frames','1000000',
                 '--snapshot-hz',str(p['snapshot_hz'])]
        if p['producer_kind']=='pc': command += ['--source',p['source'],'--fps','30']
        elif p['producer_kind']=='jetson': command += ['--camera',p['camera'],'--mode',p['camera_mode'],
              '--format',p['camera_format'],'--matrix',str(p['matrix']),
              '--width',str(p['camera_width']),'--height',str(p['camera_height'])]
        else: raise RuntimeError('producer_kind must be pc or jetson')
        if p['producer_kind']=='jetson' and p['full_range']:command += ['--full-range']
        if p['engine']: command += ['--engine',p['engine']]
        self.backend=p['backend'];self.names={}
        if p['names_json']:
            self.names=json.loads(Path(p['names_json']).read_text(encoding='utf-8')).get('names',{})
        if self.backend == 'trt' and 'person' not in self.names.values():
            raise ValueError('RT-DETR bridge requires names_json from the new reference.json')
        self.latest=LatestSlot();self.latest_snapshot=LatestSlot();self.error=None
        self.stopping=threading.Event()
        self.detection_pub=self.create_publisher(DetectionFrame,'/perception/detections',sensor_qos())
        self.snapshot_pub=self.create_publisher(Snapshot,'/perception/snapshot',sensor_qos())
        self.process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=None,text=True,
                                      encoding='utf-8',bufsize=1)
        self.reader=threading.Thread(target=self.read_lines,name='cpp-json-reader',daemon=True)
        self.reader.start()
        self.create_timer(.01,self.publish_latest)
        self.create_timer(2.0,self.report)

    def read_lines(self):
        try:
            for line in self.process.stdout:
                if self.stopping.is_set():break
                record=json.loads(line)
                if record.get('type')!='result':continue
                self.latest.put(record)
                # Keep snapshot result together, even if newer non-snapshot results arrive.
                if record.get('snapshot_ppm'):self.latest_snapshot.put(record)
            if not self.stopping.is_set(): self.error=f'C++ producer exited with code {self.process.wait()}'
        except Exception as exc:self.error=str(exc)

    def detection_message(self, record):
        result=DetectionFrame();stamp=int(record['captured_unix_ns'])
        result.header.stamp.sec=stamp//1_000_000_000;result.header.stamp.nanosec=stamp%1_000_000_000
        result.header.frame_id='camera_optical_frame'
        result.session_id=record['session_id'];result.frame_number=record['frame_id']
        result.image_width=record['width'];result.image_height=record['height']
        result.detector='cpp_'+self.backend;result.processing_ms=float(record['latency_ms'])
        result.detections=[]
        for box in record['boxes']:
            label=self.names.get(str(box['class_id']),f'class_{box["class_id"]}')
            if self.backend in ('cpu','cuda'):label='red_region_teaching_detector'
            x1,y1,x2,y2=box['xyxy']
            result.detections.append(Detection(label=label,confidence=float(box['score']),
                xmin=float(x1),ymin=float(y1),xmax=float(x2),ymax=float(y2)))
        return result

    def publish_latest(self):
        record=self.latest.take()
        if record:self.detection_pub.publish(self.detection_message(record))
        record=self.latest_snapshot.take()
        if record is None:return
        result=self.detection_message(record)
        ppm=base64.b64decode(record['snapshot_ppm'],validate=True)
        image=cv2.imdecode(np.frombuffer(ppm,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise RuntimeError('C++ PPM snapshot failed to decode')
        ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
        if not ok:raise RuntimeError('JPEG encoder failed')
        msg=Snapshot();msg.header=result.header;msg.session_id=result.session_id;msg.frame_number=result.frame_number
        msg.clock_domain='ros';msg.timestamp_source='host_acquire_system_clock'
        msg.image.header=result.header;msg.image.format='jpeg';msg.image.data=jpeg.tobytes()
        msg.original_width=result.image_width;msg.original_height=result.image_height;msg.detections=result.detections
        self.snapshot_pub.publish(msg)

    def report(self):
        if self.error:self.get_logger().error(self.error)
        self.get_logger().info(f'bridge pending replaced: detection={self.latest.replaced} snapshot={self.latest_snapshot.replaced}')

    def request_stop(self):
        self.stopping.set()
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.get_logger().error('C++ producer did not drain in 10s; terminating process, not claiming graceful GPU cleanup')
                self.process.kill();self.process.wait()
        self.reader.join(timeout=2)
        if self.process.stdout:self.process.stdout.close()


def main():spin_node(CppBridge)
