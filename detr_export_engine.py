"""Build a TensorRT engine from a static-shape ONNX model (including INT8 Q/DQ)."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import tensorrt as trt

ROOT = Path(__file__).resolve().parent


def prepare_qdq_for_tensorrt(source):
    """Avoid TensorRT's axis=1 check on scalar per-tensor Q/DQ inputs.

    ONNX ignores axis for scalar scales (per-tensor quantization). Set it to
    zero explicitly; preserve every scale, zero point and per-channel axis.
    """
    import onnx

    model = onnx.load(str(source))
    scalar_scales = {tensor.name for tensor in model.graph.initializer if len(tensor.dims) == 0}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR and len(attr.t.dims) == 0:
                    scalar_scales.update(node.output)
    changed = 0
    for node in model.graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear") or node.input[1] not in scalar_scales:
            continue
        axis = next((attr for attr in node.attribute if attr.name == "axis"), None)
        if axis is not None and axis.i == 0:
            continue
        if axis is None:
            node.attribute.append(onnx.helper.make_attribute("axis", 0))
        else:
            axis.i = 0
        changed += 1
    if not changed:
        return source
    onnx.checker.check_model(model)
    compatible = source.with_name(source.stem + ".trt.onnx")
    onnx.save(model, str(compatible))
    print(f"Set axis=0 on {changed} per-tensor Q/DQ nodes: {compatible}", flush=True)
    return compatible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=ROOT / "models/rtdetr-l-int8.onnx")
    parser.add_argument("--output", type=Path, default=ROOT / "models/rtdetr-l-int8.engine",help="Default: ONNX path with .engine extension")
    args = parser.parse_args()
    source = args.onnx.resolve()
    output = (args.output or source.with_suffix(".engine")).resolve()
    if not source.is_file():
        parser.error(f"ONNX file does not exist: {source}. Run detr_int8_quantize.py first.")
    if source == output:
        parser.error("Input ONNX and output engine must be different files.")

    source = prepare_qdq_for_tensorrt(source)
    if source == output:
        parser.error("Output engine must not overwrite the compatible ONNX file.")

    print("TensorRT version:", trt.__version__, flush=True)
    print("ONNX:", source, flush=True)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    onnx_parser = trt.OnnxParser(network, logger)
    # Parsing from a filename also resolves ONNX external weight files.
    if not onnx_parser.parse_from_file(str(source)):
        errors = "\n".join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors))
        raise RuntimeError(f"ONNX parsing failed:\n{errors}")
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print("Input:", tensor.name, tuple(tensor.shape), tensor.dtype, flush=True)
        if any(dim < 0 for dim in tensor.shape):
            raise ValueError("This script requires static input shapes; export ONNX with dynamic=False.")
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        print("Output:", tensor.name, tuple(tensor.shape), tensor.dtype, flush=True)
    if network.num_outputs == 0:
        raise RuntimeError("The parsed ONNX network has no outputs.")

    # Q/DQ in the ONNX graph specifies quantization; no additional INT8 flag is needed.
    config = builder.create_builder_config()
    # Preserve actual tensor formats for the deployment manifest/precision audit.
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed; see the TensorRT errors above.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(serialized))
    metadata = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "tensorrt": trt.__version__,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "engine": str(output),
        "engine_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "network_creation": "STRONGLY_TYPED",
        "profiling_verbosity": "DETAILED",
        "precision_note": "Types follow the ONNX graph; FP32 I/O does not imply FP32 internal computation.",
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Saved engine:", output, flush=True)


if __name__ == "__main__":
    main()
