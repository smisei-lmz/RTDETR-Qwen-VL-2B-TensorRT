import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Export RT-DETR to a static FP32 ONNX model")
    parser.add_argument("--model", type=Path, default=ROOT / "rtdetr-l.pt", help="RT-DETR .pt checkpoint")
    parser.add_argument("--output", type=Path, help="Output path; defaults to weights/<model-name>.onnx")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset; RT-DETR requires at least 16")
    args = parser.parse_args()
    if args.opset < 16:
        parser.error("RT-DETR ONNX export requires --opset >= 16")
    dst = (args.output or ROOT / "weights" / f"{args.model.stem}.onnx").resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT / "ultralytics"))
    from ultralytics import RTDETR

    model = RTDETR(args.model)
    src = Path(model.export(
        format="onnx", imgsz=640, batch=1, dynamic=False,
        simplify=False, opset=args.opset, device="cpu",
    )).resolve()
    if src != dst:
        shutil.move(str(src), str(dst))
    print("Saved ONNX to:", dst)


if __name__ == "__main__":
    main()
