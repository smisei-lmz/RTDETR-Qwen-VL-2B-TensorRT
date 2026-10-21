#!/usr/bin/env python3
"""Check one image with the SAME VLM backend before involving ROS2."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ros_ws' / 'src' / 'edge_assistant'))
from edge_assistant.backends import make_backend
from edge_assistant.core import FrameKey, Snapshot, Request, Status, run_request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', type=Path)
    p.add_argument('--question', default='请描述图像中的物体、颜色和相对位置。')
    p.add_argument('--backend', choices=['mock', 'qwen2_vl'], default='mock')
    p.add_argument('--model', default='Qwen/Qwen2-VL-2B-Instruct')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--timeout', type=float, default=60)
    args = p.parse_args()
    if args.image.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
        p.error('provide a JPEG or PNG file')
    if not 0 < args.timeout <= 120:
        p.error('timeout must be 0..120 seconds')
    data = args.image.read_bytes()
    if not 0 < len(data) <= 1024 * 1024:
        p.error('resize/re-encode the image to at most 1 MiB first')
    backend = make_backend(args.backend, args.model, args.device, 0.1)
    # Time is the file request time, not the original photograph exposure time.
    snapshot = Snapshot(FrameKey(str(uuid.uuid4()), 1), time.time_ns(), data,
                        'image/png' if args.image.suffix.lower() == '.png' else 'image/jpeg',
                        timestamp_source='file_request_time')
    req = Request(str(uuid.uuid4()), args.question, snapshot,
                  timeout_s=args.timeout, max_answer_age_s=min(180, args.timeout + 10))
    answer = run_request(req, backend, lambda: False, time.time_ns)
    print(json.dumps(asdict(answer), ensure_ascii=False, indent=2))
    return 0 if answer.status == Status.OK else 2


if __name__ == '__main__':
    raise SystemExit(main())
