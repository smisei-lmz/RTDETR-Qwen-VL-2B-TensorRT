import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NAMES = ("person", "bicycle", "car", "motorcycle", "airplane")


def prepare_dataset(coco, dataset):
    """Create a five-class dataset from COCO images and YOLO detection labels."""
    for split in ("train2017", "val2017"):
        for folder in ("images", "labels"):
            source = coco / folder / split
            if not source.is_dir():
                raise FileNotFoundError(f"Missing {source}; provide COCO images and YOLO detection labels first.")
    if dataset.exists():
        raise FileExistsError(f"{dataset} already exists; choose a new --dataset directory.")

    for source_split, target_split in (("train2017", "train"), ("val2017", "val")):
        images = dataset / "images" / target_split
        labels = dataset / "labels" / target_split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        count = 0
        for image in sorted((coco / "images" / source_split).glob("*.jpg")):
            label = coco / "labels" / source_split / f"{image.stem}.txt"
            rows = []
            if label.exists():
                for line in label.read_text().splitlines():
                    fields = line.split()
                    if not fields:
                        continue
                    if int(fields[0]) not in range(len(NAMES)):
                        continue
                    if len(fields) == 5:
                        rows.append(line)
                    elif len(fields) >= 7 and len(fields) % 2 == 1:
                        # Convert COCO polygon labels to normalized xywh boxes.
                        xs = [float(x) for x in fields[1::2]]
                        ys = [float(y) for y in fields[2::2]]
                        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
                        box = ((xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin)
                        rows.append(fields[0] + " " + " ".join(f"{v:.8g}" for v in box))
                    else:
                        raise ValueError(f"{label}: expected a YOLO box or polygon label.")
            # Keep images without selected objects as background examples.
            (images / image.name).symlink_to(image)
            (labels / label.name).write_text("\n".join(rows) + ("\n" if rows else ""))
            count += 1
        if not count:
            raise ValueError(f"No JPG images in {coco / 'images' / source_split}")
        print(f"{target_split}: {count} images")

    data = dataset / "data.yaml"
    class_names = "\n".join(f"  {i}: {name}" for i, name in enumerate(NAMES))
    data.write_text(f"path: {json.dumps(dataset.as_posix(), ensure_ascii=False)}\ntrain: images/train\nval: images/val\nnames:\n{class_names}\n")
    return data


def main():
    parser = argparse.ArgumentParser(description="Train RT-DETR on the first five COCO classes")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset" / "coco5")
    parser.add_argument("--coco", type=Path, help="COCO root with images/ and YOLO-format labels/")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare labels without starting training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="GPU index or cpu")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    data = dataset / "data.yaml"
    if args.coco:
        data = prepare_dataset(args.coco.resolve(), dataset)
        print(f"Dataset ready: {data}")
    elif args.prepare_only:
        parser.error("--prepare-only requires --coco")
    if args.prepare_only:
        return
    if not data.is_file():
        parser.error(f"Missing {data}; run with --coco /path/to/coco --prepare-only first.")

    sys.path.insert(0, str(ROOT / "ultralytics"))
    from ultralytics import RTDETR

    model = RTDETR(ROOT / "rtdetr-l.pt")
    model.train(
        data=str(data), epochs=args.epochs, imgsz=640, batch=args.batch,
        device=args.device, deterministic=False,
        project=str(ROOT / "runs" / "detect"), name="coco5",
    )


if __name__ == "__main__":
    main()
