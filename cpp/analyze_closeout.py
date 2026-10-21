"""Check closeout evidence and export a compact JSON report and figure."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sample_identities(path, count, images=None):
    seen = set()
    with path.open() as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            latency = float(row["latency_ms"])
            if not math.isfinite(latency) or latency < 0:
                raise ValueError("Invalid latency sample")
            if images is not None:
                if int(row["frame_id"]) != index or int(row["image_index"]) != index % images:
                    raise ValueError("Stability sample identity mismatch")
            else:
                key = (int(row["round"]), int(row["lanes"]), int(row["frame_id"]))
                if key in seen:
                    raise ValueError("Duplicate benchmark sample identity")
                seen.add(key)
    actual = index + 1 if "index" in locals() else 0
    if actual != count:
        raise ValueError(f"Expected {count} samples, got {actual}")


def main(directory, waive_long_stability=False):
    quality = load(directory / "quality.json")
    lock = load(directory / "deployment.lock.json")
    stability = load(directory / "stability.json")
    correctness = load(directory / "correctness.json")
    engines = load(directory / "engines.json")
    selected = lock["engine_key"]
    preprocessing = load(directory / f"raw/preprocess_{selected}.summary.json")
    lanes = load(directory / f"raw/lane_{selected}.summary.json")
    sample_identities(directory / f"raw/lane_{selected}.samples.csv",
                      sum(run["completed"] for run in lanes["runs"]))
    sample_identities(directory / "raw/stability.samples.csv", stability["completed"],
                      stability["image_count"])
    expected = {(run["round"], run["lanes"], i) for run in lanes["runs"]
                for i in range(lanes["frames_per_round"])}
    with (directory / f"raw/lane_{selected}.samples.csv").open() as stream:
        actual = {(int(row["round"]), int(row["lanes"]), int(row["frame_id"]))
                  for row in csv.DictReader(stream)}
    if actual != expected:
        raise ValueError("Benchmark frames do not match the declared rounds")
    engine_hash = hashlib.sha256((ROOT / lock["engine"]).read_bytes()).hexdigest()
    quality_hash = hashlib.sha256((directory / "quality.json").read_bytes()).hexdigest()
    criteria = {
        "selected_engine_quality_budget": quality["results"][lock["engine_key"]]["within_budget"],
        "artifact_hashes": engine_hash == lock["engine_sha256"] == stability["engine_sha256"]
            == engines[selected]["engine_sha256"] and quality_hash == lock["quality_report_sha256"],
        "preprocess_parity": preprocessing["input_max_abs"] <= 0.002
            and preprocessing["output_max_abs"] <= 1e-5
            and preprocessing["postprocess_box_max_abs"] <= 1e-3,
        "lane_lifecycle": all(run["passed"] and run["exit_code"] == 0 for run in correctness["runs"])
            and any(run["engine_key"] == selected and run["engine_sha256"] == engine_hash
                    for run in correctness["runs"]),
        "short_stability_check": stability["passed"],
        "long_stability_requirement": waive_long_stability or stability["plan_30_minute_requirement_met"],
        "all_submitted_frames_collected": stability["submitted"] == stability["completed"]
            and stability["failed"] == 0 and stability["all_frames_collected"],
        "lane_destroyed": stability["lane_destroyed"],
        "sample_identities": True,
    }
    interrupted_path = directory / "stability_interrupted.json"
    interrupted = load(interrupted_path) if interrupted_path.exists() else None
    memory = interrupted["memory_samples"] if interrupted else stability["memory_samples"]
    tail = memory[len(memory) // 2:]
    mx = sum(row["elapsed_s"] for row in tail) / len(tail)
    my = sum(row["rss_bytes"] for row in tail) / len(tail)
    denominator = sum((row["elapsed_s"] - mx) ** 2 for row in tail)
    slope = (sum((row["elapsed_s"] - mx) * (row["rss_bytes"] - my) for row in tail) / denominator
             if denominator else None)
    report = {
        "scope": "PC DETR-only deployment closeout; fixed-image performance and continuous two-image stability",
        "passed": all(criteria.values()), "criteria": criteria,
        "long_stability": {"required": not waive_long_stability,
                           "waived_at_user_request": waive_long_stability,
                           "continuous_30_minutes_completed": stability["plan_30_minute_requirement_met"]},
        "interrupted_run": {key: value for key, value in interrupted.items() if key != "memory_samples"}
            if interrupted else None,
        "deployment": lock, "quality": quality,
        "preprocessing": preprocessing, "lane_benchmark": lanes,
        "stability": {key: value for key, value in stability.items() if key != "memory_samples"},
        "memory_observation": {
            "second_half_rss_slope_kib_per_minute": slope * 60 / 1024 if slope is not None else None,
            "rss_end_minus_start_bytes": stability["rss_after_bytes"] - stability["rss_before_bytes"],
            "trend_source": "interrupted long-run telemetry" if interrupted else "completed stability run",
            "observed_rss_range_bytes": max(row["rss_bytes"] for row in memory) - min(row["rss_bytes"] for row in memory),
            "observed_device_free_range_bytes": max(row["device_free_bytes"] for row in memory) - min(row["device_free_bytes"] for row in memory),
            "device_free_after_lane_teardown_minus_before_bytes":
                stability["device_free_after_lane_teardown_bytes"] - stability["device_free_after_bytes"],
            "note": "RSS belongs to the C++ process. cudaMemGetInfo observes the entire GPU. CUDA module/cache memory can remain while the engine/process lives.",
        },
        "remaining_system_validation": ["labeled workstation videos and event behavior", "ROS integration", "VLM co-running", "Jetson"],
    }
    save(directory / "preprocessing.json", preprocessing)
    save(directory / "lane_benchmark.json", lanes)
    save(directory / "closeout.json", report)

    cache = directory / "raw/matplotlib_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ["#70849c", "#168a78", "#c77948"]
    names = ["FP32", "Mixed FP16", "INT8 Add6"]
    aps = [quality["results"][key]["map50_95"] * 100 for key in ("fp32", "fp16", "int8_add6")]
    axes[0, 0].bar(names, aps, color=colors)
    axes[0, 0].set(ylabel="COCO mAP50-95 (%)", ylim=(0, max(60, max(aps) + 5)),
                   title=f"Same C++ pipeline, {quality['image_count']} labeled images")
    for i, value in enumerate(aps):
        axes[0, 0].text(i, value + 0.7, f"{value:.2f}", ha="center")
    mean = [preprocessing[mode]["total"]["mean_ms"] for mode in ("cpu", "cuda")]
    axes[0, 1].bar(["CPU pre/post", "CUDA pre/post"], mean, color=colors[:2])
    axes[0, 1].set(ylabel="Mean chain latency (ms)", title=f"{selected}: fixed-image pre/post comparison")
    for i, value in enumerate(mean):
        axes[0, 1].text(i, value + 0.1, f"{value:.2f}", ha="center")
    runs = sorted(lanes["aggregate"], key=lambda run: run["lanes"])
    axes[1, 0].bar(["1 lane", "2 lanes"], [r["fps"] for r in runs], color=colors[:2])
    axes[1, 0].set(ylabel="Completed frames / second", title=f"{selected}: closed-loop throughput and P99")
    latency_axis = axes[1, 0].twinx()
    latency_axis.plot([0, 1], [r["latency_p99_ms"] for r in runs], "o-", color="#bc4b51")
    latency_axis.set_ylabel("P99 latency (ms)", color="#bc4b51")
    minutes = [row["elapsed_s"] / 60 for row in memory]
    axes[1, 1].plot(minutes, [row["rss_bytes"] / 1048576 for row in memory], color=colors[1])
    axes[1, 1].set(xlabel="Continuous inference (minutes)", ylabel="C++ process RSS (MiB)",
                   title=(f"Long run interrupted: {interrupted['last_telemetry_elapsed_s']:.0f}s logged"
                          if interrupted else f"Stability: {stability['completed']:,} completed frames"))
    device_axis = axes[1, 1].twinx()
    device_axis.plot(minutes, [row["device_free_bytes"] / 1048576 for row in memory], color=colors[0], alpha=0.6)
    device_axis.set_ylabel("Whole-device free memory (MiB)", color=colors[0])
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    figure.suptitle("RT-DETR-L deployment closeout | RTX 5060 | TensorRT 10", fontsize=14)
    figure.savefig(directory / "overview.png", dpi=160)
    plt.close(figure)
    print(json.dumps({"passed": report["passed"], "criteria": criteria,
                      "memory_observation": report["memory_observation"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/detr_closeout_20261019")
    parser.add_argument("--waive-long-stability", action="store_true",
                        help="Record that the user waived the 30-minute requirement; do not claim it passed")
    args = parser.parse_args()
    main(args.directory.resolve(), args.waive_long_stability)
