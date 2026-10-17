from pathlib import Path

import cv2
import numpy as np


CALIB_DIR = Path(
    "/home/smisei/project/Yolov11-ONNX-TensorRT/ultralytics_yolo11-main/datasets/coco/images/calibration"
)

OUTPUT_PATH = Path(__file__).resolve().parent / "calibration.npy"

INPUT_H = 640
INPUT_W = 640

MAX_IMAGES = 100

def preprocess(
    image
):

    image = cv2.resize(image, (INPUT_W, INPUT_H))


    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    image = np.ascontiguousarray(
        image,
        dtype=np.float32
    )

    image /= 255.0

    return image


def main():

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    paths = sorted(
        p
        for p in CALIB_DIR.rglob("*")
        if p.suffix.lower()
        in extensions
    )

    if not paths:

        raise RuntimeError(
            "No calibration images found."
        )

    paths = paths[
        :MAX_IMAGES
    ]

    print(
        "Calibration images:",
        len(paths)
    )

    calibration_data = []


    for i, path in enumerate(
        paths
    ):

        image = cv2.imread(
            str(path)
        )

        if image is None:

            print(
                "skip:",
                path
            )

            continue

        tensor = preprocess(
            image
        )

        calibration_data.append(
            tensor
        )

        if (
            (i + 1) % 50
            == 0
        ):

            print(
                f"{i + 1}/"
                f"{len(paths)}"
            )


    calibration_data = np.stack(
        calibration_data,
        axis=0
    )


    # 最终：
    # [N,3,640,640]

    print(
        "\n===== Calibration ====="
    )

    print(
        "shape:",
        calibration_data.shape
    )

    print(
        "dtype:",
        calibration_data.dtype
    )

    print(
        "min:",
        calibration_data.min()
    )

    print(
        "max:",
        calibration_data.max()
    )

    print(
        "mean:",
        calibration_data.mean()
    )


    np.save(
        OUTPUT_PATH,
        calibration_data
    )


    print(
        "\nsaved:",
        OUTPUT_PATH
    )

if __name__ == "__main__":
    main()