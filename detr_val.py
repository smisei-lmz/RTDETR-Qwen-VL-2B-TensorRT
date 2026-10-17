import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Validate pretrained RT-DETR on a labeled dataset")
    parser.add_argument("--model", default=str(ROOT / "rtdetr-l.pt"))
    parser.add_argument("--data", default=str(ROOT / "coco_val.yaml"))
    parser.add_argument("--device", default="0", help="GPU index or cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="val_pretrained", help="Validation result directory name")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "ultralytics"))
    from ultralytics import RTDETR

    model = RTDETR(args.model)
    metrics = model.val(
        data=args.data,
        split="val",
        imgsz=640,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
    )
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
