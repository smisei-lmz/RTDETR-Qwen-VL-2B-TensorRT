"""Compare dumped C++ tensors and detections with Python using the SAME engine."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prefix", required=True, type=Path)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError("Cannot deserialize engine; use the project's TensorRT 10 Python")
    context = engine.create_execution_context()
    inputs = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
              if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
    outputs = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
               if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
    input_name, output_name = inputs[0], outputs[0]
    shape = tuple(engine.get_tensor_shape(input_name))
    output_shape = tuple(engine.get_tensor_shape(output_name))
    cpp_input = np.fromfile(str(args.prefix) + ".input.bin", dtype=np.float32).reshape(shape)
    cpp_output = np.fromfile(str(args.prefix) + ".output.bin", dtype=np.float32).reshape(output_shape)
    resized = cv2.resize(image, (shape[3], shape[2]), interpolation=cv2.INTER_LINEAR)
    python_input = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

    def infer(array):
        device_input = torch.from_numpy(array).to("cuda")
        device_output = torch.empty(output_shape, dtype=torch.float32, device="cuda")
        context.set_tensor_address(input_name, device_input.data_ptr())
        context.set_tensor_address(output_name, device_output.data_ptr())
        if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT inference failed")
        torch.cuda.synchronize()
        return device_output.cpu().numpy()

    same_input_output = infer(cpp_input)
    python_output = infer(python_input)
    result = {
        "tensorrt": trt.__version__,
        "input_max_abs": float(np.abs(cpp_input - python_input).max()),
        "output_same_input_max_abs": float(np.abs(cpp_output - same_input_output).max()),
        "output_python_preprocess_max_abs": float(np.abs(cpp_output - python_output).max()),
    }
    np.testing.assert_allclose(cpp_input, python_input, rtol=0, atol=1e-7)
    np.testing.assert_allclose(cpp_output, same_input_output, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cpp_output, python_output, rtol=1e-5, atol=1e-5)
    # Decode independently in NumPy to verify row layout and original-image scaling.
    pred = cpp_output[0]
    pred = pred[pred[:, 4] >= 0.25]
    boxes = np.empty_like(pred[:, :4])
    boxes[:, :2] = (pred[:, :2] - pred[:, 2:4] / 2) * [image.shape[1], image.shape[0]]
    boxes[:, 2:4] = (pred[:, :2] + pred[:, 2:4] / 2) * [image.shape[1], image.shape[0]]
    expected = np.column_stack([pred[:, 5], pred[:, 4], boxes])
    with open(str(args.prefix) + ".detections.csv") as file:
        reader = csv.reader(file)
        next(reader)
        actual = np.array([[float(value) for value in row] for row in reader], dtype=np.float64).reshape(-1, 6)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-4)
    box_error = float(np.abs(actual[:, 2:] - boxes).max()) if len(pred) else 0.0
    result.update(detections=len(pred), decoded_boxes_max_abs=box_error, passed=True)
    print(json.dumps(result, indent=2))
    Path(str(args.prefix) + ".verification.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
