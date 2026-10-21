"""Reproduce DETR deployment quality, manifests and continuous-lane stability.

Run with .venv-trt10/bin/python. GPU work is sequential; do not benchmark other
GPU jobs concurrently. Large raw predictions/logs live in output/raw/.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def command_info(command):
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    return {"command": command, "exit_code": result.returncode, "output": result.stdout.strip()}


def environment(config_path):
    import tensorrt as trt
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "tensorrt": trt.__version__,
        "pycocotools": importlib.metadata.version("pycocotools"),
        "config_sha256": digest(config_path),
        "git": command_info(["git", "rev-parse", "HEAD"]),
        "git_status": command_info(["git", "status", "--short"]),
        "gpu": command_info(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                             "--format=csv,noheader"]),
        "cuda": command_info(["/usr/local/cuda-13.4/bin/nvcc", "--version"]),
        "source_sha256": {name: digest(ROOT / name) for name in (
            "cpp/closeout.py", "cpp/lane_validate.cpp", "cpp/src/laneManager.cpp",
            "cpp/include/laneManager.h", "cpp/src/CudaPreprocess.cu", "cpp/src/CudaPostprocess.cu")},
        "binary_sha256": digest(ROOT / "cpp/build-trt10/trt_lane_validate"),
    }


def manifest(config, raw):
    import onnx
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    records = {}
    for name, relative in config["engines"].items():
        path = ROOT / relative
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Cannot load {relative} with TRT {trt.__version__}")
        tensors = []
        for i in range(engine.num_io_tensors):
            tensor = engine.get_tensor_name(i)
            tensors.append({"name": tensor, "mode": str(engine.get_tensor_mode(tensor)),
                            "shape": list(engine.get_tensor_shape(tensor)),
                            "dtype": str(engine.get_tensor_dtype(tensor)),
                            "format": engine.get_tensor_format_desc(tensor)})
        inspector = engine.create_engine_inspector()
        details = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        save(raw / f"{name}.layers.json", details)
        formats = Counter()

        def visit(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "Format/Datatype":
                        formats[str(child)] += 1
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(details)
        graph_path = ROOT / config["onnx"][name]
        graph = onnx.load(str(graph_path))
        record = {"engine": relative, "engine_sha256": digest(path), "engine_bytes": path.stat().st_size,
                  "onnx": config["onnx"][name], "onnx_sha256": digest(graph_path),
                  "io": tensors, "layer_tensor_formats": dict(formats),
                  "onnx_initializer_types": dict(Counter(onnx.TensorProto.DataType.Name(t.data_type)
                                                         for t in graph.graph.initializer)),
                  "precision_note": "Initializer and inspected tensor formats are evidence of mixed types; not per-instruction arithmetic precision."}
        provenance = path.with_suffix(path.suffix + ".json")
        if provenance.exists():
            record["build"] = json.loads(provenance.read_text())
        else:
            record["build"] = {"status": "historical build provenance unavailable; source association declared by config"}
        records[name] = record
        del inspector, engine
    records["checkpoint"] = {"path": "models/rtdetr-l.pt", "sha256": digest(ROOT / "models/rtdetr-l.pt")}
    return records


def dataset(config, output, count):
    annotations = ROOT / config["quality"]["annotations"]
    content = json.loads(annotations.read_text())
    directory = ROOT / config["quality"]["images"]
    selected = sorted((image for image in content["images"]
                       if (directory / image["file_name"]).is_file()), key=lambda image: image["id"])
    if count:
        selected = selected[:count]
    if not selected:
        raise ValueError("No labeled images available")
    records = []
    paths = []
    for image in selected:
        path = directory / image["file_name"]
        paths.append(str(path))
        records.append({**image, "sha256": digest(path)})
    selection = {"annotations": config["quality"]["annotations"], "annotations_sha256": digest(annotations),
                 "available_labeled_count": len([i for i in content["images"]
                                                  if (directory / i["file_name"]).is_file()]),
                 "official_image_count": len(content["images"]), "evaluated_count": len(selected),
                 "selection": "all locally available labeled images sorted by COCO id; optional first-N smoke run",
                 "images": records}
    save(output / "dataset.json", selection)
    image_list = output / "images.txt"
    image_list.write_text("\n".join(paths) + "\n")
    return selected, image_list


def execute(command, log):
    print("RUN", " ".join(map(str, command)), flush=True)
    with log.open("w") as stream:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def quality(config, output, count):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    save(output / "engines.json", manifest(config, raw))
    selected, image_list = dataset(config, output, count)
    ground_truth = COCO(str(ROOT / config["quality"]["annotations"]))
    categories = sorted(ground_truth.getCatIds())
    if len(categories) != 80:
        raise ValueError("Expected COCO80 category mapping")
    results = {}
    for name, engine in config["engines"].items():
        prefix = raw / name
        execute([ROOT / "cpp/build-trt10/trt_lane_validate", "--engine", ROOT / engine,
                 "--images", image_list, "--prefix", prefix,
                 "--threshold", config["quality"]["evaluation_threshold"]], raw / f"{name}.log")
        run = json.loads(prefix.with_suffix(".summary.json").read_text())
        if run["completed"] != len(selected) or not run["passed"]:
            raise RuntimeError(f"Incomplete C++ dataset run: {name}")
        # Every image, including zero-detection images, remains in COCOeval.imgIds.
        predictions = []
        with prefix.with_suffix(".predictions.csv").open() as stream:
            for row in csv.DictReader(stream):
                image = selected[int(row["image_index"])]
                x1, y1, x2, y2 = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
                x1, x2 = [min(image["width"], max(0.0, x)) for x in (x1, x2)]
                y1, y2 = [min(image["height"], max(0.0, y)) for y in (y1, y2)]
                if x2 <= x1 or y2 <= y1:
                    continue
                predictions.append({"image_id": image["id"],
                                    "category_id": categories[int(row["class_id"])],
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(row["score"])})
        if not predictions:
            raise RuntimeError(f"No predictions from {name}")
        save(raw / f"{name}.coco.json", predictions)
        evaluator = COCOeval(ground_truth, ground_truth.loadRes(predictions), "bbox")
        evaluator.params.imgIds = [image["id"] for image in selected]
        evaluator.params.maxDets = [1, 10, config["quality"]["max_detections"]]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        results[name] = {"map50_95": float(evaluator.stats[0]), "map50": float(evaluator.stats[1]),
                         "map75": float(evaluator.stats[2]), "coco_stats": evaluator.stats.tolist(),
                         "evaluated_images": len(selected), "prediction_count": len(predictions), "cpp_run": run}
        save(output / "quality.partial.json", results)
    reference = results["fp32"]["map50_95"]
    if reference <= 0:
        raise RuntimeError("FP32 reference has no measurable detection quality")
    budget = config["quality"]["max_map_drop_absolute"]
    for name, result in results.items():
        result["map_drop_from_fp32"] = reference - result["map50_95"]
        result["within_budget"] = result["map_drop_from_fp32"] <= budget
    preferred = config["preferred_deployment"]
    selected_engine = preferred if results[preferred]["within_budget"] else "fp32"
    report = {"scope": "actual C++ CUDA preprocessing + TensorRT10 + CUDA decode; COCO bbox evaluation",
              "image_count": len(selected), "official_image_count": len(ground_truth.imgs),
              "full_official_validation_set": len(selected) == len(ground_truth.imgs),
              "dataset_sha256": digest(output / "dataset.json"),
              "protocol": {**config["quality"], "clip_boxes_to_image": True,
                           "extra_nms": False, "same_backend": "TensorRT 10"},
              "results": results, "selected_deployment": selected_engine,
              "selection_reason": "Prefer the planned mixed-FP16 baseline if within the predeclared 1pp budget; otherwise retain FP32. INT8 remains an experiment."}
    save(output / "quality.json", report)
    save(output / "deployment.lock.json", {
        "model_family": config["model_family"],
        "engine_key": selected_engine, "engine": config["engines"][selected_engine],
        "engine_sha256": digest(ROOT / config["engines"][selected_engine]),
        "input_shape": config["input_shape"], "output_shape": config["output_shape"],
        "io_dtype": config["io_dtype"], "preprocess": config["preprocess"],
        "postprocess": config["postprocess"], "runtime": config["runtime"],
        "quality_report_sha256": digest(output / "quality.json"),
        "scope": "PC configuration; Jetson requires a device-local engine rebuild and validation",
    })
    print(json.dumps({"selected_deployment": selected_engine,
                      "map50_95": {k: v["map50_95"] for k, v in results.items()}}, indent=2), flush=True)


def summarize_stability(output, seconds):
    prefix = output / "raw/stability"
    run = json.loads(prefix.with_suffix(".summary.json").read_text())
    with prefix.with_suffix(".samples.csv").open() as stream:
        values = sorted(float(row["latency_ms"]) for row in csv.DictReader(stream))
    with prefix.with_suffix(".memory.csv").open() as stream:
        memory = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    def percentile(p):
        return values[math.ceil(p * len(values)) - 1]
    tail = memory[len(memory) // 2:]
    rss_range = max(row["rss_bytes"] for row in tail) - min(row["rss_bytes"] for row in tail)
    gpu_range = max(row["device_free_bytes"] for row in tail) - min(row["device_free_bytes"] for row in tail)
    report = {**run, "target_seconds": seconds,
              "duration_met": run["elapsed_s"] >= seconds,
              "sample_count_matches": len(values) == run["completed"],
              "latency_mean_ms": sum(values) / len(values),
              "latency_p50_ms": percentile(0.5), "latency_p95_ms": percentile(0.95),
              "latency_p99_ms": percentile(0.99), "percentile_method": "nearest_rank",
              "memory_samples": memory, "second_half_rss_range_bytes": rss_range,
              "second_half_device_free_range_bytes": gpu_range,
              "memory_note": "RSS is this C++ process; device free memory covers the whole GPU. Ranges alone do not prove absence of all leaks.",
              "scope": "closed-loop two preloaded images, repeated-output equality each frame; no video arrival queue or ROS"}
    report["passed"] = bool(run["passed"] and report["duration_met"] and report["sample_count_matches"])
    save(output / "stability.json", report)
    return report


def frozen_deployment(config, output, config_path):
    quality_path = output / "quality.json"
    selected = json.loads(quality_path.read_text())["selected_deployment"]
    lock = json.loads((output / "deployment.lock.json").read_text())
    evaluated_environment = json.loads((output / "environment.json").read_text())
    engine_path = ROOT / config["engines"][selected]
    if (digest(config_path) != evaluated_environment["config_sha256"] or
            lock["quality_report_sha256"] != digest(quality_path) or
            lock["engine_key"] != selected or digest(engine_path) != lock["engine_sha256"]):
        raise ValueError("Configuration, quality report or engine changed after evaluation; rerun evaluate")
    return selected, lock, engine_path


def checks(config, output, config_path):
    selected, lock, engine_path = frozen_deployment(config, output, config_path)
    if config["runtime"] != {"lanes": 1, "confidence_threshold": 0.25, "num_classes": 80, "poll_us": 50}:
        raise ValueError("This benchmark protocol requires the fixed COCO80/0.25/50us runtime configuration")
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    save(output / "checks.environment.json", environment(config_path))
    runs = []
    record = {"scope": "GPU C++ lane configuration, CPU parity, ownership, ROI, state gating and teardown",
              "test_source_sha256": digest(ROOT / "cpp/lane_check.cpp"),
              "test_binary_sha256": digest(ROOT / "cpp/build-trt10/trt_lane_check"), "runs": runs}
    save(output / "correctness.json", record)
    for key in dict.fromkeys([selected, "int8_add6"]):
        log = raw / f"{key}.final_lane_check.log"
        command = [ROOT / "cpp/build-trt10/trt_lane_check", ROOT / config["engines"][key],
                   *[ROOT / image for image in config["stability_images"]]]
        execute(command, log)
        if "PASS:" not in log.read_text():
            raise RuntimeError("Correctness check did not report completion")
        runs.append({"engine_key": key, "engine_sha256": digest(ROOT / config["engines"][key]),
                     "command": list(map(str, command)), "exit_code": 0, "passed": True,
                     "log": str(log.relative_to(output))})
        save(output / "correctness.json", record)
    image = ROOT / config["stability_images"][0]
    execute([ROOT / "cpp/build-trt10/trt_benchmark", engine_path, image, raw / f"preprocess_{selected}",
             "--iterations", 200, "--warmup", 30, "--opencv-threads", 1], raw / f"preprocess_{selected}.log")
    execute([ROOT / "cpp/build-trt10/trt_lane_benchmark", engine_path, image, raw / f"lane_{selected}",
             "--frames", 500, "--warmup", 30, "--rounds", 5, "--lanes", "both", "--poll-us", 50],
            raw / f"lane_{selected}.log")
    for target, source in (("preprocessing", f"preprocess_{selected}"), ("lane_benchmark", f"lane_{selected}")):
        save(output / f"{target}.json", json.loads((raw / f"{source}.summary.json").read_text()))
    if digest(engine_path) != lock["engine_sha256"]:
        raise ValueError("Engine changed during checks")


def stability(config, output, seconds, config_path):
    quality_path = output / "quality.json"
    if not quality_path.exists():
        raise ValueError("Run evaluate first to select a quality-validated engine")
    selected, lock, engine_path = frozen_deployment(config, output, config_path)
    paths = [ROOT / path for path in config["stability_images"]]
    image_hashes = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    image_list = output / "stability_images.txt"
    image_list.write_text("\n".join(map(str, paths)) + "\n")
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    save(output / "stability.environment.json", environment(config_path))
    execute([ROOT / "cpp/build-trt10/trt_lane_validate", "--engine", ROOT / config["engines"][selected],
             "--images", image_list, "--prefix", raw / "stability",
             "--threshold", config["runtime"]["confidence_threshold"],
             "--warmup", config["stability"]["warmup"], "--duration-seconds", seconds], raw / "stability.log")
    report = summarize_stability(output, seconds)
    report["engine_key"] = selected
    if digest(engine_path) != lock["engine_sha256"] or any(
            digest(ROOT / image["path"]) != image["sha256"] for image in image_hashes):
        raise ValueError("Engine or input image files changed during stability run")
    report["engine_sha256"] = lock["engine_sha256"]
    report["images"] = image_hashes
    report["plan_30_minute_requirement_met"] = report["passed"] and seconds >= 1800
    save(output / "stability.json", report)
    print(json.dumps({k: report[k] for k in ("passed", "completed", "elapsed_s", "latency_p99_ms")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["evaluate", "checks", "stability"])
    parser.add_argument("--config", type=Path, default=ROOT / "cpp/config/detr_deployment.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/detr_closeout_20261019")
    parser.add_argument("--count", type=int, default=0, help="0=all available labeled images; otherwise first N for smoke testing")
    parser.add_argument("--seconds", type=int, default=5,
                        help="Stability duration, default 5 seconds; explicitly pass 1800 for the optional long run")
    args = parser.parse_args()
    if args.count < 0 or (args.seconds is not None and args.seconds <= 0):
        parser.error("Invalid count or duration")
    args_config_path = args.config.resolve()
    config = json.loads(args_config_path.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "evaluate":
        save(output / "environment.json", environment(args_config_path))
        quality(config, output, args.count)
    elif args.command == "checks":
        checks(config, output, args_config_path)
    else:
        stability(config, output, args.seconds or config["stability"]["duration_seconds"], args_config_path)
