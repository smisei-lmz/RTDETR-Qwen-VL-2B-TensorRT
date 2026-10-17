import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ultralytics"))

from ultralytics import RTDETR


def main():
    parser = argparse.ArgumentParser(description="Run RT-DETR inference with Ultralytics")
    parser.add_argument("source", help="Image, directory, video, stream URL, or webcam index")
    parser.add_argument("--model", default=str(ROOT / "rtdetr-l.pt"), help="RT-DETR .pt checkpoint")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    model = RTDETR(args.model)
    for result in model.predict(source=args.source, imgsz=640, conf=args.conf, save=True, stream=True):
        print(f"{result.path}: {len(result.boxes)} detections")


if __name__ == "__main__":
    main()
