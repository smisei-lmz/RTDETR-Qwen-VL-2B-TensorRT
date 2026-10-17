import argparse
from collections import Counter
from pathlib import Path

import numpy as np

ONNX_PATH = "/home/smisei/project/RTDETR-ONNX-TensorRT/models/rtdetr-l-fp32.onnx"

CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.npy"

OUTPUT_PATH = Path(__file__).resolve().parent / "models/rtdetr-l-int8.onnx"
BOX_ADD_PATTERN = r"^/model\.28/decoder/Add(?:_[1-5])?$"
BOX_ADD_NAMES = {"/model.28/decoder/Add"} | {f"/model.28/decoder/Add_{i}" for i in range(1, 6)}


def check_box_add_names(path):
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    found = {n.name for n in model.graph.node if n.op_type == "Add" and n.name in BOX_ADD_NAMES}
    if found != BOX_ADD_NAMES:
        raise ValueError(f"Expected six box-update Add nodes; missing: {sorted(BOX_ADD_NAMES - found)}")


def protect_box_add(source, output):
    """Restore only these consumers; keep shared Q/DQ for other consumers."""
    import onnx
    from detr_quant_sensitivity import ablate

    check_box_add_names(source)
    temporary = output.with_name(output.stem + ".protected.tmp.onnx")
    try:
        restored = ablate(source, temporary, restore_pattern=BOX_ADD_PATTERN)
        model = onnx.load(str(temporary), load_external_data=False)
        producers = {o: n for n in model.graph.node for o in n.output}
        for node in model.graph.node:
            if node.name in BOX_ADD_NAMES:
                if any(i in producers and producers[i].op_type == "DequantizeLinear" for i in node.input):
                    raise RuntimeError(f"Box Add still has quantized input: {node.name}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Protected six box-update Add nodes; bypassed Q/DQ inputs on {len(restored)} nodes.", flush=True)

# Supported INT8 operator types in the installed ModelOpt ONNX backend.
SUPPORTED_OPS = [
    "Add", "AveragePool", "BatchNormalization", "Clip", "Conv", "ConvTranspose",
    "Gemm", "GlobalAveragePool", "MatMul", "MaxPool", "Mul",
]


def report_coverage(path):
    """Report graph Q/DQ placement, not TensorRT kernel execution precision."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    producers = {output: node for node in model.graph.node for output in node.output}
    totals = Counter()
    with_dq = Counter()
    for node in model.graph.node:
        if node.op_type in SUPPORTED_OPS:
            totals[node.op_type] += 1
            if any(producers.get(name) is not None and
                   producers[name].op_type == "DequantizeLinear" for name in node.input):
                with_dq[node.op_type] += 1
    print("\nONNX placement: nodes with >=1 DQ input / total nodes", flush=True)
    for op in sorted(totals):
        print(f"  {op}: {with_dq[op]}/{totals[op]}", flush=True)
    print("This counts graph placement; actual engine precision depends on TensorRT fusion.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Quantize supported RT-DETR operators, including MatMul/Gemm")
    parser.add_argument("--onnx", type=Path, default=Path(ONNX_PATH))
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--output", type=Path, help="Default: models/rtdetr-l-int8[-box-add6-fp32].onnx")
    parser.add_argument("--protect-box-add", action="store_true",
                        help="Exclude only the six decoder box-coordinate Add nodes")
    parser.add_argument("--restore-box-add-from", type=Path,
                        help="Protect these Add nodes in an existing quantized ONNX without recalibration")
    parser.add_argument("--op-types", nargs="+", choices=SUPPORTED_OPS, default=SUPPORTED_OPS,
                        help="Default: all supported types; ModelOpt may exclude individual nodes")
    parser.add_argument("--inspect", type=Path, help="Only report Q/DQ placement in an existing ONNX")
    args = parser.parse_args()
    if args.inspect:
        report_coverage(args.inspect)
        return
    if args.restore_box_add_from:
        args.protect_box_add = True
    if args.output is None:
        args.output = (OUTPUT_PATH.with_name("rtdetr-l-int8-box-add6-fp32.onnx")
                       if args.protect_box_add else OUTPUT_PATH)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    source = args.restore_box_add_from or args.onnx
    if source.resolve() == args.output.resolve():
        parser.error("Input ONNX and output ONNX must be different files")
    if args.restore_box_add_from:
        protect_box_add(source, args.output)
        print("Output:", args.output)
        report_coverage(args.output)
        return
    assert args.onnx.exists(), args.onnx
    assert args.calibration.exists(), args.calibration
    if args.protect_box_add:
        check_box_add_names(args.onnx)

    from modelopt.onnx.quantization import quantize

    print("Loading calibration data...")

    calibration_data = np.load(
        args.calibration, mmap_mode="r"
    )

    print(
        "Calibration:",
        calibration_data.shape,
        calibration_data.dtype
    )

    print()
    print("Start INT8 PTQ...")
    print("Requested operator types:", ", ".join(args.op_types), flush=True)
    print("Unsupported or excluded nodes retain FP32.", flush=True)
    print()

    quantize(
        onnx_path=str(args.onnx),

        # INT8 PTQ
        quantize_mode="int8",
        op_types_to_quantize=args.op_types,
        nodes_to_exclude=[BOX_ADD_PATTERN] if args.protect_box_add else None,

        # [N,3,640,640]
        calibration_data=calibration_data,

        # 激活值 calibration 算法
        calibration_method="entropy",

        # 优先使用GPU
        calibration_eps=[
            "cuda:0",
            "cpu"
        ],

        # Preserve FP32 for unquantized operations; the FP16 path lost accuracy.
        high_precision_dtype="fp32",

        output_path=str(args.output),

        log_level="INFO"
    )

    if args.protect_box_add:
        # Also bypass any input Q/DQ shared with other quantized consumers.
        protect_box_add(args.output, args.output)

    print()
    print("==============================")
    print("INT8 PTQ finished")
    print("==============================")
    print("Input :", args.onnx)
    print("Output:", args.output)
    report_coverage(args.output)


if __name__ == "__main__":
    main()
