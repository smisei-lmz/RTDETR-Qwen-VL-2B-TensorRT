import os
import onnx
from modelopt.onnx.autocast import convert_to_mixed_precision

# 配置路径
INPUT_ONNX = "/home/smisei/project/RTDETR-ONNX-TensorRT/weights/rtdetr-l-fp32.onnx"
OUTPUT_ONNX = "/home/smisei/project/RTDETR-ONNX-TensorRT/weights/rtdetr-l-fp32mix16.onnx"

def main():
    # 1. 检查输入文件是否存在
    if not os.path.exists(INPUT_ONNX):
        raise FileNotFoundError(f"Input ONNX model not found: {INPUT_ONNX}")

    print(f"Loading model from: {INPUT_ONNX}")
    
    # 2. 执行混合精度转换
    # keep_io_types=True 确保输入/输出张量仍为 FP32，避免推理时数据预处理报错
    try:
        model_fp16 = convert_to_mixed_precision(
            onnx_path=INPUT_ONNX, 
            low_precision_type="fp16",
            keep_io_types=True
        )
        print("Mixed precision conversion completed.")
    except Exception as e:
        print(f"Error during conversion: {e}")
        return

    # 3. 保存模型
    onnx.save(model_fp16, OUTPUT_ONNX)
    print(f"Model saved to: {OUTPUT_ONNX}")

    # 4. 验证模型完整性
    try:
        model = onnx.load(OUTPUT_ONNX)
        onnx.checker.check_model(model)
        print("\nONNX checker: PASS")
    except onnx.checker.ValidationError as e:
        print(f"\nONNX checker: FAILED - {e}")
        return

    # 5. 打印文件大小对比
    fp32_size = os.path.getsize(INPUT_ONNX) / 1024 / 1024
    fp16_size = os.path.getsize(OUTPUT_ONNX) / 1024 / 1024

    print(f"FP32 ONNX Size: {fp32_size:.2f} MB")
    print(f"Mixed FP16 ONNX Size: {fp16_size:.2f} MB")
    print(f"Size Reduction: {(1 - fp16_size/fp32_size)*100:.1f}%")

if __name__ == "__main__":
    main()
