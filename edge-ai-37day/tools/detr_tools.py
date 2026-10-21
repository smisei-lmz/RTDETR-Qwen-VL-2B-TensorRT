"""Reference, ONNX, raw TensorRT plan and COCO evaluation for RT-DETRv2-S."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ros_ws/src/edge_assistant'))
from edge_assistant.rtdetr_backend import (
    RTDetrBackend, MODEL_ID, CONTRACT, INPUT_NAME, OUTPUT_NAME, decode_rows)

MODELS = Path(os.environ.get('EDGE_MODEL_DIR', str(ROOT/'artifacts/models')))
REFERENCE = ROOT/'data/reference'


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def image_path(image):
    path = Path(image).expanduser().resolve() if image else ROOT/'data/workstation/reference.jpg'
    if not path.is_file():
        raise RuntimeError('Provide --image PATH to a real workstation photograph (person/objects visible)')
    return path


def reference(args):
    import torch
    import transformers
    from PIL import Image, ImageDraw
    model = RTDetrBackend(args.weights)
    source = image_path(args.image)
    with Image.open(source) as opened:
        image = opened.convert('RGB')
    MODELS.mkdir(parents=True, exist_ok=True)
    REFERENCE.mkdir(parents=True, exist_ok=True)
    image.save(REFERENCE/'reference.ppm')
    thumbnail = image.copy()
    thumbnail.thumbnail((640, 640))
    thumbnail.save(MODELS/'vlm-input.jpg', quality=85)
    for _ in range(5):
        model.predict(image)
    torch.cuda.synchronize()
    started = time.perf_counter()
    boxes = model.predict(image)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter()-started)*1000
    if not any(b['score'] >= .35 for b in boxes):
        raise RuntimeError('No confident reference boxes; use another real photograph')
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box['xyxy'], outline='red', width=2)
        draw.text(box['xyxy'][:2], model.names[box['class_id']], fill='red')
    image.save(MODELS/'reference.jpg')
    save_json(MODELS/'reference.json', dict(
        boxes=boxes, names=model.names, width=image.width, height=image.height,
        latency_ms=elapsed, model=args.weights, contract=CONTRACT,
        torch=torch.__version__, transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(0), image=str(source),
        preprocess='bilinear stretch, half-pixel, no antialias, RGB /255; no letterbox',
        note='One sample, not a performance benchmark; dataset evaluation required'))
    print(f'RT-DETR reference: {len(boxes)} boxes; {elapsed:.2f} ms')


def export_onnx(args):
    import torch
    import onnx
    import onnxruntime as ort
    import numpy as np
    from PIL import Image
    model = RTDetrBackend(args.weights)
    with Image.open(image_path(args.image)) as image:
        sample = model.tensor(image.convert('RGB'))
    module = model.export_module()
    MODELS.mkdir(parents=True, exist_ok=True)
    dest = MODELS/'detector.onnx'
    with torch.inference_mode():
        expected = module(sample).cpu().numpy()
        torch.onnx.export(module, sample, str(dest), input_names=[INPUT_NAME],
                          output_names=[OUTPUT_NAME], opset_version=17,
                          dynamic_axes=None, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(str(dest)))
    session = ort.InferenceSession(str(dest), providers=['CPUExecutionProvider'])
    actual = session.run([OUTPUT_NAME], {INPUT_NAME: sample.cpu().numpy()})[0]
    if actual.shape != (1, 300, 6):
        raise RuntimeError(f'Unexpected deployment output {actual.shape}')
    # Compare detections rather than row order: top-k ordering can change near ties.
    report = compare_boxes(decode_rows(expected[0], 640, 640), decode_rows(actual[0], 640, 640))
    save_json(MODELS/'onnx-parity.json', report)
    save_json(MODELS/'onnx_manifest.json', dict(
        contract=CONTRACT, model=args.weights, input=[1, 3, 640, 640], output=[1, 300, 6],
        names=model.names, sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        onnx=onnx.__version__, onnxruntime=ort.__version__))
    print(dest)


def build_engine(args):
    import tensorrt as trt
    # Always regenerate: never build an old detector graph under a new model name.
    export_onnx(args)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse((MODELS/'detector.onnx').read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT build failed; inspect parser/operators/precision logs')
    raw = bytes(serialized)
    (MODELS/'detector.plan').write_bytes(raw)
    save_json(MODELS/'engine_manifest.json', dict(
        contract=CONTRACT, model=args.weights, tensorrt=trt.__version__,
        raw_plan_sha256=hashlib.sha256(raw).hexdigest(),
        input='images [1,3,640,640] RGB bilinear stretch /255',
        output='rtdetr_detections [1,300,6]: normalized xyxy, score, class_id; no NMS',
        precision='FP16 allowed, FP32 I/O',
        note='Raw plan, no metadata prefix. Rebuild separately on Jetson.'))
    print(MODELS/'detector.plan')


class PlanRunner:
    """Named TensorRT I/O, using torch-owned CUDA buffers and one CUDA stream."""
    def __init__(self, path):
        import torch
        import tensorrt as trt
        self.torch = torch
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError('Cannot deserialize raw RT-DETR plan')
        if self.engine.num_io_tensors != 2:
            raise ValueError('Require one RT-DETR input and one output')
        expected = {INPUT_NAME: (1, 3, 640, 640), OUTPUT_NAME: (1, 300, 6)}
        self.context = self.engine.create_execution_context()
        self.buffers = {}
        for name, shape in expected.items():
            if tuple(self.engine.get_tensor_shape(name)) != shape:
                raise ValueError('Engine does not implement ' + CONTRACT)
            dtype = self.engine.get_tensor_dtype(name)
            if dtype not in (trt.float32, trt.float16):
                raise ValueError('Require FP32/FP16 engine I/O')
            buffer = torch.empty(shape, device='cuda:0', dtype=torch.float32 if dtype == trt.float32 else torch.float16)
            self.buffers[name] = buffer
            self.context.set_tensor_address(name, buffer.data_ptr())

    def predict_tensor(self, tensor, width, height, threshold=.25):
        stream = self.torch.cuda.current_stream()
        self.buffers[INPUT_NAME].copy_(tensor)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT enqueue failed')
        stream.synchronize()
        return decode_rows(self.buffers[OUTPUT_NAME][0].float().cpu().numpy(), width, height, threshold)


def box_iou(a, b):
    a, b = a['xyxy'], b['xyxy']
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return area / max(1e-12, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def compare_boxes(reference, actual):
    reference = [b for b in reference if b['score'] >= .35]
    actual = [b for b in actual if b['score'] >= .35]
    if not reference:
        raise RuntimeError('No confident reference boxes')
    used, matches = set(), []
    for ref in reference:
        candidates = [(box_iou(ref, box), i, box) for i, box in enumerate(actual)
                      if i not in used and box['class_id'] == ref['class_id']]
        if not candidates:
            raise RuntimeError(f'Missing class {ref["class_id"]}')
        overlap, i, candidate = max(candidates, key=lambda row: row[0])
        if overlap < .90 or abs(ref['score']-candidate['score']) > .08:
            raise RuntimeError(f'Parity failed: IoU={overlap:.4f}; inspect preprocessing/export')
        used.add(i)
        matches.append(dict(class_id=ref['class_id'], iou=overlap,
                            score_error=abs(ref['score']-candidate['score'])))
    if len(used) != len(actual):
        raise RuntimeError('Unexpected extra confident detections')
    return dict(pass_=True, matches=matches, threshold=.35,
                limitation='Single-image deployment parity, not dataset accuracy')


def compare(args):
    data = json.loads((MODELS/'reference.json').read_text(encoding='utf-8'))
    if data.get('contract') != CONTRACT:
        raise ValueError('Old reference artifact: rerun Day03 for RT-DETR')
    if args.events:
        records = [json.loads(line) for line in Path(args.events).read_text(encoding='utf-8').splitlines() if line.strip()]
        results = [r for r in records if r.get('type') == 'result']
        if not results:
            raise RuntimeError('No C++ detection result')
        actual = results[0]['boxes']
    else:
        from PIL import Image
        model = RTDetrBackend(args.weights)
        runner = PlanRunner(MODELS/'detector.plan')
        with Image.open(REFERENCE/'reference.ppm') as image:
            actual = runner.predict_tensor(model.tensor(image.convert('RGB')), image.width, image.height)
    report = compare_boxes(data['boxes'], actual)
    save_json(Path(args.report) if args.report else MODELS/'parity.json', report)
    print(json.dumps(report))


def evaluate(args):
    """Explicit local COCO dataset manifest; no silent download or fabricated mAP."""
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    path = Path(args.dataset).resolve()
    settings = json.loads(path.read_text(encoding='utf-8'))
    image_dir = (path.parent/settings['images']).resolve()
    coco = COCO(str((path.parent/settings['annotations']).resolve()))
    model = RTDetrBackend(args.weights)
    runner = PlanRunner(MODELS/'detector.plan')
    categories = {c['name']: c['id'] for c in coco.dataset['categories']}
    mapping = {i: categories[name] for i, name in model.names.items() if name in categories}
    if set(mapping.values()) != set(categories.values()):
        raise ValueError('Dataset categories must use supported COCO class names')
    predictions = {'pytorch': [], 'tensorrt': []}
    ids = sorted(coco.getImgIds())
    if not ids:
        raise ValueError('Dataset contains no images')
    for image_id in ids:
        with Image.open(image_dir/coco.imgs[image_id]['file_name']) as opened:
            image = opened.convert('RGB')
        results = {'pytorch': model.predict(image, .001),
                   'tensorrt': runner.predict_tensor(model.tensor(image), image.width, image.height, .001)}
        for runtime, boxes in results.items():
            for box in boxes:
                if box['class_id'] not in mapping:
                    continue
                x1, y1, x2, y2 = box['xyxy']
                predictions[runtime].append(dict(image_id=image_id, category_id=mapping[box['class_id']],
                    bbox=[x1, y1, x2-x1, y2-y1], score=box['score']))
    reports = dict(dataset=str(path), images=len(ids), contract=CONTRACT)
    for runtime, rows in predictions.items():
        if not rows:
            raise RuntimeError(runtime + ': no detections; inspect data/model before evaluation')
        coco.dataset.setdefault('info', {})
        evaluator = COCOeval(coco, coco.loadRes(rows), 'bbox')
        evaluator.params.imgIds = ids
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        reports[runtime] = dict(mAP50_95=float(evaluator.stats[0]), mAP50=float(evaluator.stats[1]))
    reports['mAP50_95_drop'] = reports['pytorch']['mAP50_95'] - reports['tensorrt']['mAP50_95']
    save_json(MODELS/'accuracy.json', reports)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['reference', 'onnx', 'engine', 'compare', 'evaluate'])
    parser.add_argument('--weights', default=MODEL_ID)
    parser.add_argument('--image')
    parser.add_argument('--events')
    parser.add_argument('--report')
    parser.add_argument('--dataset', default='data/coco_eval.json')
    args = parser.parse_args()
    {'reference': reference, 'onnx': export_onnx, 'engine': build_engine,
     'compare': compare, 'evaluate': evaluate}[args.task](args)


if __name__ == '__main__':
    main()
