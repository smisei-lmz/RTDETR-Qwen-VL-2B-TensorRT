"""Reproduce optimization calculations and extract the two steady-state nsys traces."""
import bisect
import collections
import csv
import json
import math
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PREFIX = RESULTS / "lane_benchmark_20261017"


def load(path):
    return json.loads(path.read_text())


def percentile(values, p):
    values = sorted(values)
    return values[math.ceil(len(values) * p) - 1]


def covered_time(intervals):
    """Return union duration and duration with >=2 distinct owners active."""
    changes = []
    for start, end, owner in intervals:
        if end > start:
            changes.extend([(start, owner, 1), (end, owner, -1)])
    changes.sort()
    counts = collections.Counter()
    active_owners = 0
    union = overlap = 0
    previous = changes[0][0] if changes else 0
    for time, owner, delta in changes:
        if active_owners:
            union += time - previous
        if active_owners >= 2:
            overlap += time - previous
        was_active = counts[owner] > 0
        counts[owner] += delta
        active_owners += int(counts[owner] > 0) - int(was_active)
        previous = time
    return union, overlap


def trace_summary(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    ranges = db.execute("""
        SELECT n.start,n.end,COALESCE(n.text,s.value),n.globalTid
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
        WHERE COALESCE(n.text,s.value)='lane.submit' ORDER BY n.start
    """).fetchall()
    starts = [r[0] for r in ranges]
    start, end = db.execute("""
        SELECT n.start,n.end FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON s.id=n.textId
        WHERE COALESCE(n.text,s.value) LIKE 'steady_lane%'
    """).fetchone()
    apis = {
        row[0]: row[1:] for row in db.execute(
            "SELECT correlationId,start,end,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    }

    def submission(correlation):
        api = apis.get(correlation)
        if not api:
            return None
        index = bisect.bisect_right(starts, api[0]) - 1
        if index >= 0 and api[0] <= ranges[index][1] and api[2] == ranges[index][3]:
            return index
        return None

    copies = db.execute("""
        SELECT start,end,streamId,correlationId,bytes
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
    """).fetchall()
    # The source image is 640*480*3 bytes, so these copies identify main lane streams.
    owners = {}
    for a, b, stream, correlation, size in copies:
        index = submission(correlation)
        if size == 640 * 480 * 3 and index is not None:
            owners[index] = stream
    kernels = db.execute("""
        SELECT start,end,streamId,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL
    """).fetchall()
    kernel_intervals = []
    associated = []
    post_api_gaps = []
    for a, b, stream, correlation in kernels:
        a, b = max(start, a), min(end, b)
        kernel_intervals.append((a, b, 0))
        owner = owners.get(submission(correlation))
        if owner is not None:
            associated.append((a, b, owner))
        api = apis.get(correlation)
        if api:
            post_api_gaps.append(max(0, a - api[1]) / 1e6)
    all_intervals = list(kernel_intervals)
    for a, b, stream, correlation, size in copies:
        a, b = max(start, a), min(end, b)
        all_intervals.append((a, b, 0))
    kernel_union, _ = covered_time(kernel_intervals)
    work_union, _ = covered_time(all_intervals)
    _, cross_lane_overlap = covered_time(associated)
    cpu_ranges = dict(db.execute("""
        SELECT COALESCE(n.text,s.value),SUM(n.end-n.start)/1e6
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
        WHERE COALESCE(n.text,s.value) LIKE 'lane.%'
        GROUP BY COALESCE(n.text,s.value)
    """))
    api_totals = db.execute("""
        SELECT s.value,COUNT(*),SUM(r.end-r.start)/1e6
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
        GROUP BY s.value ORDER BY SUM(r.end-r.start) DESC LIMIT 8
    """).fetchall()
    allocation_calls = db.execute("""
        SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id=r.nameId
        WHERE s.value LIKE '%Malloc%' OR s.value LIKE '%Free%'
    """).fetchone()[0]
    top_kernels = db.execute("""
        SELECT s.value,COUNT(*),SUM(k.end-k.start)/1e6
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
        GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC LIMIT 8
    """).fetchall()
    db.close()
    return {
        "path": str(path.relative_to(ROOT)),
        "captured_submissions": len(ranges),
        "main_lane_stream_ids": sorted(set(owners.values())),
        "steady_range_ms": (end - start) / 1e6,
        "kernel_instances": len(kernels),
        "kernels_per_frame": len(kernels) / len(ranges),
        "kernel_union_ms": kernel_union / 1e6,
        "kernel_active_wall_percent": 100 * kernel_union / (end - start),
        "kernel_or_copy_active_wall_percent": 100 * work_union / (end - start),
        "cross_lane_kernel_overlap_ms": cross_lane_overlap / 1e6,
        "cross_lane_kernel_overlap_wall_percent": 100 * cross_lane_overlap / (end - start),
        "kernels_associated_with_lane": len(associated),
        "positive_post_api_gap_p50_ms": percentile(post_api_gaps, 0.50),
        "positive_post_api_gap_p95_ms": percentile(post_api_gaps, 0.95),
        "cpu_nvtx_ranges_ms": cpu_ranges,
        "steady_cuda_allocation_or_free_calls": allocation_calls,
        "largest_runtime_api_totals": [
            {"name": name, "calls": count, "total_ms": duration}
            for name, count, duration in api_totals
        ],
        "largest_kernel_groups": [
            {"name": name, "instances": count, "summed_gpu_ms": duration}
            for name, count, duration in top_kernels
        ],
        "scope": "Profiler timings are diagnostic only. Kernel coverage is not SM utilization. "
                 "Cross-lane overlap groups auxiliary kernels by the enclosing host submission "
                 "and its H2D main stream; it does not count same-lane auxiliary overlap.",
    }


def main():
    baseline = load(RESULTS / "benchmark_single_lane.summary.json")
    current = load(PREFIX.with_suffix(".summary.json"))
    environment = load(PREFIX.with_suffix(".environment.json"))
    original_environment = load(RESULTS / "benchmark_single_lane.environment.json")
    if environment["engine_sha256"] != original_environment["engine_sha256"]:
        raise ValueError("Historical and current engines differ; do not combine conclusions")

    samples = list(csv.DictReader(PREFIX.with_suffix(".samples.csv").open()))
    expected = current["frames_per_round"] * current["rounds"] * 2
    if len(samples) != expected:
        raise ValueError("Incomplete measurement samples")
    for lanes in (1, 2):
        for round_id in range(current["rounds"]):
            group = [s for s in samples if int(s["lanes"]) == lanes and int(s["round"]) == round_id]
            if (len(group) != current["frames_per_round"] or
                    {int(s["frame_id"]) for s in group} != set(range(current["frames_per_round"]))):
                raise ValueError("Missing or duplicated frame identities")

    cpu, gpu = baseline["cpu"], baseline["cuda"]
    reduction = lambda before, after: 100 * (1 - after / before)
    first = load(ROOT / "runs/quant_sensitivity/summary.json")
    second = load(ROOT / "runs/quant_sensitivity_add/summary.json")
    original_int8 = next(r for r in first if r["name"] == "reference_int8")["map"]
    protected = next(r for r in second if r["name"] == "add_box_refinement")["map"]
    lanes = {}
    for row in current["aggregate"]:
        count = row["lanes"]
        per_round = [r["fps"] for r in current["runs"] if r["lanes"] == count]
        lanes[count] = {
            **row,
            "fps_per_round_min": min(per_round),
            "fps_per_round_max": max(per_round),
            "process_cpu_percent": 100 * row["process_cpu_ms"] / row["wall_ms"],
        }
    analysis = {
        "historical_cpu_vs_cuda": {
            "latency_reduction_percent": reduction(cpu["total"]["mean_ms"], gpu["total"]["mean_ms"]),
            "latency_speedup": cpu["total"]["mean_ms"] / gpu["total"]["mean_ms"],
            "p95_reduction_percent": reduction(cpu["total"]["p95_ms"], gpu["total"]["p95_ms"]),
            "h2d_byte_reduction_percent": reduction(baseline["h2d_cpu_bytes"], baseline["h2d_cuda_bytes"]),
            "h2d_time_reduction_percent": reduction(cpu["h2d"]["mean_ms"], gpu["h2d"]["mean_ms"]),
            "scope": "One existing paired 200-frame fixed-image trial; input 640x480 BGR, "
                     "network 640x640, batch=1; excludes decode and initialization. "
                     "Combined preprocessing/transfer change, not isolated kernel-fusion benefit.",
        },
        "lane_benchmark": {
            "lanes": lanes,
            "two_vs_one_fps_change_percent": 100 * (lanes[2]["fps"] / lanes[1]["fps"] - 1),
            "two_vs_one_mean_latency_change_percent":
                100 * (lanes[2]["latency_mean_ms"] / lanes[1]["latency_mean_ms"] - 1),
            "one_vs_two_p99_reduction_percent":
                reduction(lanes[2]["latency_p99_ms"], lanes[1]["latency_p99_ms"]),
            "scope": current["scope"],
        },
        "quantization_same_subset": {
            "images": 200, "backend": "ONNX Runtime CUDA", "restored_nodes": 6,
            "before_map_percent": original_int8 * 100,
            "after_map_percent": protected * 100,
            "improvement_percentage_points": (protected - original_int8) * 100,
            "scope": "Fixed calibration scales, same image subset and backend; "
                     "not a full-validation-set or current-engine mAP claim.",
        },
        "traces": [
            trace_summary(RESULTS / f"profiles/lane{i}_20261017.sqlite") for i in (1, 2)
        ],
    }
    output = RESULTS / "optimization_analysis_20261017.json"
    output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")

    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS / "profiles/matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = ["#496B9C", "#2B9A83"]
    for offset, key, label in [(-0.18, "mean_ms", "Mean"), (0.18, "p95_ms", "P95")]:
        values = [baseline[m]["total"][key] for m in ("cpu", "cuda")]
        bars = axes[0].bar([offset, 1 + offset], values, width=0.34, label=label)
        axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[0].set_xticks([0, 1], ["CPU pre/post", "CUDA pre/post"])
    axes[0].set_ylabel("Pipeline latency (ms)")
    axes[0].set_title("Existing paired trial (200 frames/mode)")
    axes[0].legend(frameon=False)
    axes[0].set_ylim(0, 12)

    for round_id in range(current["rounds"]):
        values = [next(r["fps"] for r in current["runs"]
                       if r["round"] == round_id and r["lanes"] == n) for n in (1, 2)]
        axes[1].plot([1, 2], values, "o-", alpha=0.75, label=f"Round {round_id + 1}")
    axes[1].set_xticks([1, 2], ["1 lane", "2 lanes"])
    axes[1].set_ylabel("Completed frames / wall time (FPS)")
    axes[1].set_title("New 5-round trial (500 frames/round)")
    axes[1].legend(frameon=False, fontsize=8)

    labels = ["P50", "P95", "P99"]
    for index, n in enumerate((1, 2)):
        values = [lanes[n][f"latency_{p.lower()}_ms"] for p in labels]
        bars = axes[2].bar([i + (index - 0.5) * 0.36 for i in range(3)], values,
                           width=0.34, color=colors[index], label=f"{n} lane(s)")
        axes[2].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[2].set_xticks(range(3), labels)
    axes[2].set_ylabel("Submit-to-collected-result latency (ms)")
    axes[2].set_title("New trial: 2,500 samples/config")
    axes[2].legend(frameon=False)
    axes[2].set_ylim(0, 19)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle("RT-DETR-L / RTX 5060 / batch=1 / fixed image; profiler-free performance",
                 fontsize=12)
    fig.tight_layout()
    figure = RESULTS / "optimization_analysis_20261017.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    print("Saved", output.relative_to(ROOT), "and", figure.relative_to(ROOT))
    for trace in analysis["traces"]:
        print(json.dumps({k: trace[k] for k in [
            "main_lane_stream_ids", "steady_range_ms", "kernels_per_frame",
            "kernel_active_wall_percent", "cross_lane_kernel_overlap_wall_percent",
            "kernels_associated_with_lane", "steady_cuda_allocation_or_free_calls",
            "cpu_nvtx_ranges_ms"]}, indent=2))


if __name__ == "__main__":
    main()
