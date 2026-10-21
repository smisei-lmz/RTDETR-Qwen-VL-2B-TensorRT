"""RT-DETRv2-S adapter shared by conversion tools and the ROS reference node.

Deployment contract: RGB bilinear stretch to 640x640, float NCHW /255;
one output [1,300,6]: normalized xyxy, sigmoid score, contiguous class id.
Top-k is over query/class pairs, as in focal-loss RT-DETR postprocessing.
No NMS. Imports of GPU libraries are deliberately lazy.
"""
import math

MODEL_ID = 'PekingU/rtdetr_v2_r18vd'
CONTRACT = 'rtdetr_v2_topk_v1'
INPUT_NAME = 'images'
OUTPUT_NAME = 'rtdetr_detections'


def decode_rows(rows, width, height, threshold=.25):
    """Decode exported rows; also used to verify exported runtimes independently."""
    if width <= 0 or height <= 0 or not 0 <= threshold <= 1:
        raise ValueError('invalid image size or threshold')
    boxes = []
    for row in rows:
        if len(row) != 6:
            raise ValueError('RT-DETR output row must contain six values')
        x1, y1, x2, y2, score, label = map(float, row)
        if not all(math.isfinite(v) for v in row):
            raise ValueError('non-finite RT-DETR output')
        if not 0 <= score <= 1 or label != int(label) or not 0 <= label < 80:
            raise ValueError('invalid RT-DETR score/class; incompatible export')
        if score < threshold:
            continue
        xyxy = [max(0., min(1., x1))*width, max(0., min(1., y1))*height,
                max(0., min(1., x2))*width, max(0., min(1., y2))*height]
        if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
            boxes.append(dict(class_id=int(label), score=score, xyxy=xyxy))
    return boxes


class RTDetrBackend:
    def __init__(self, weights=MODEL_ID, device='cuda:0'):
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
        if str(device).isdigit():
            device = 'cuda:' + str(device)
        self.torch, self.device = torch, device
        self.processor = RTDetrImageProcessor.from_pretrained(weights)
        if self.processor.do_normalize or not self.processor.do_rescale:
            raise ValueError('This adapter requires the RT-DETR /255, no-mean/std contract')
        if abs(self.processor.rescale_factor - 1/255) > 1e-8:
            raise ValueError('unsupported rescale factor')
        self.model = RTDetrV2ForObjectDetection.from_pretrained(weights).to(device).eval()
        if self.model.config.num_labels != 80 or not self.model.config.use_focal_loss:
            raise ValueError('Require COCO80 RT-DETRv2 focal-loss weights')
        aliases = {'motorbike': 'motorcycle', 'aeroplane': 'airplane', 'sofa': 'couch',
                   'pottedplant': 'potted plant', 'diningtable': 'dining table', 'tvmonitor': 'tv'}
        self.names = {int(k): aliases.get(v, v) for k, v in self.model.config.id2label.items()}

    def tensor(self, rgb):
        import numpy as np
        pixels = self.torch.from_numpy(np.asarray(rgb, dtype=np.uint8).copy())
        pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        # Match the CUDA half-pixel bilinear implementation. PIL downsampling
        # may antialias differently; this is an explicit deployment convention.
        return self.torch.nn.functional.interpolate(
            pixels, size=(640, 640), mode='bilinear', align_corners=False) / 255.

    def export_module(self):
        torch = self.torch
        class Deployment(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, images):
                output = self.model(pixel_values=images)
                scores, indices = output.logits.sigmoid().flatten(1).topk(300, dim=1)
                query_indices = indices // 80
                labels = indices % 80
                boxes = output.pred_boxes.gather(1, query_indices.unsqueeze(-1).expand(-1, -1, 4))
                center, size = boxes[..., :2], boxes[..., 2:]
                xyxy = torch.cat((center - size / 2, center + size / 2), dim=-1)
                return torch.cat((xyxy, scores.unsqueeze(-1), labels.to(xyxy.dtype).unsqueeze(-1)), dim=-1)
        return Deployment(self.model).eval()

    def predict(self, rgb, threshold=.25):
        import numpy as np
        array = np.asarray(rgb)
        with self.torch.inference_mode():
            rows = self.export_module()(self.tensor(array))[0].cpu().numpy()
        return decode_rows(rows, array.shape[1], array.shape[0], threshold)
