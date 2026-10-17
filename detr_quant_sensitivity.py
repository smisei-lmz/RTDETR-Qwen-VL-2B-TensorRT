"""Controlled Q/DQ ablations with fixed scales and a reproducible val subset.

Restoring a consumer reconnects its inputs to the floating inputs of Q->DQ.
Other consumers retain their Q/DQ and calibration scales. This diagnoses an
existing quantized graph; it does not recalibrate or promise engine precision.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def ablate(source, output, restore_ops=(), restore_pattern=None, keep_ops=None):
    import onnx

    model = onnx.load(str(source))
    producers = {o: n for n in model.graph.node for o in n.output}
    restored = []
    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            continue
        selected = (node.op_type in restore_ops or
                    (restore_pattern is not None and re.search(restore_pattern, node.name)) or
                    (keep_ops is not None and node.op_type not in keep_ops))
        if not selected:
            continue
        changed = False
        for index, name in enumerate(node.input):
            dq = producers.get(name)
            if dq is None or dq.op_type != "DequantizeLinear":
                continue
            q = producers.get(dq.input[0])
            if q is None or q.op_type != "QuantizeLinear":
                raise ValueError(f"Cannot restore {node.name}: DQ has no floating Q input")
            node.input[index] = q.input[0]
            changed = True
        if changed:
            restored.append({"name": node.name, "op": node.op_type})

    # Remove Q/DQ nodes that no longer have a path to an output.
    live = {o.name for o in model.graph.output}
    nodes = []
    for node in reversed(model.graph.node):
        if any(o in live for o in node.output):
            nodes.append(node)
            live.update(node.input)
    del model.graph.node[:]
    model.graph.node.extend(reversed(nodes))
    initializers = [t for t in model.graph.initializer if t.name in live]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    onnx.checker.check_model(model)
    onnx.save(model, str(output))
    return restored


def worker(args):
    sys.path.insert(0, str(ROOT / "ultralytics"))
    from ultralytics import RTDETR

    metrics = RTDETR(str(args.model)).val(
        data=str(args.data), imgsz=640, batch=1, device="0", workers=0,
        plots=False, verbose=False, project=str(args.directory / "val"),
        name=args.label, exist_ok=True,
    )
    result = dict(map=float(metrics.box.map), map50=float(metrics.box.map50),
                  precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                  speed=metrics.speed)
    args.result.write_text(json.dumps(result, indent=2))


def make_subset(directory, count):
    import yaml

    images = sorted((ROOT / "ultralytics/coco/images/val2017").glob("*.jpg"))
    selected = sorted(random.Random(42).sample(images, min(count, len(images))))
    if not selected:
        raise ValueError("No val2017 images found")
    image_list = directory / f"val_{len(selected)}_seed42.txt"
    image_list.write_text("\n".join(str(p) for p in selected) + "\n")
    config = yaml.safe_load((ROOT / "coco_val.yaml").read_text())
    config["val"] = str(image_list)
    config.pop("download", None)
    data = directory / f"val_{len(selected)}_seed42.yaml"
    data.write_text(yaml.safe_dump(config, allow_unicode=True))
    return data, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "models/rtdetr-l-int8.onnx")
    parser.add_argument("--float-model", type=Path, default=ROOT / "models/rtdetr-l-fp32.onnx")
    parser.add_argument("--directory", type=Path, default=ROOT / "runs/quant_sensitivity")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--cases", type=Path, help="JSON: list of {name, restore_ops/restore_pattern/keep_ops}")
    parser.add_argument("--keep-models", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--data", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.directory = args.directory.resolve()
    args.directory.mkdir(parents=True, exist_ok=True)
    data, images = make_subset(args.directory, args.count)
    cases = (json.loads(args.cases.read_text()) if args.cases else [
        {"name": "reference_fp32", "reference": "fp32"},
        {"name": "reference_int8", "reference": "int8"},
        {"name": "restore_MatMul", "restore_ops": ["MatMul"]},
        {"name": "restore_Gemm", "restore_ops": ["Gemm"]},
        {"name": "restore_Add", "restore_ops": ["Add"]},
        {"name": "restore_Mul", "restore_ops": ["Mul"]},
        {"name": "restore_Clip", "restore_ops": ["Clip"]},
        {"name": "keep_Conv_MatMul_Gemm", "keep_ops": ["Conv", "MatMul", "Gemm"]},
    ])
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    manifest = {"source": str(args.source.resolve()), "source_sha256": source_hash,
                "count": len(images), "seed": 42, "cases": cases,
                "note": "Fixed original calibration scales. A diagnostic ablation, not recalibration."}
    (args.directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    results = []
    for case in cases:
        name = case["name"]
        result_file = args.directory / f"{name}.json"
        output = args.directory / f"{name}.onnx"
        restored = []
        if case.get("reference"):
            model = args.float_model if case["reference"] == "fp32" else args.source
        else:
            restored = ablate(args.source, output, case.get("restore_ops", []),
                              case.get("restore_pattern"), case.get("keep_ops"))
            model = output
        print(f"START {name}: restored {len(restored)} nodes {dict(Counter(n['op'] for n in restored))}", flush=True)
        (args.directory / f"{name}_nodes.json").write_text(json.dumps(restored, indent=2))
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--model", str(model), "--data", str(data), "--directory", str(args.directory),
                   "--label", name, "--result", str(result_file)]
        with (args.directory / f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(result_file.read_text())
        result.update(name=name, restored_nodes=len(restored))
        results.append(result)
        (args.directory / "summary.json").write_text(json.dumps(results, indent=2))
        print(f"DONE {name}: mAP={result['map']:.5f}, AP50={result['map50']:.5f}", flush=True)
        if not args.keep_models and not case.get("reference"):
            output.unlink()


if __name__ == "__main__":
    main()
