"""Causal, batch-one inference and confidence-first box decoding."""
from types import MethodType
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from .common import seed_everything
from .data import inference_clip
from .model import build_model, load_weights


def _sparse_forward(self, current_feat, ref_feat=None, domain_ids=None):
    if self.training or current_feat.shape[0] != 1:
        raise ValueError('Sparse expert execution requires eval mode and batch size one')
    current_pool = F.adaptive_avg_pool2d(current_feat, 1).flatten(1)
    ref_pool = (F.adaptive_avg_pool2d(ref_feat, 1).flatten(1)
                if ref_feat is not None else torch.zeros_like(current_pool))
    probs = self.gate_network(torch.cat([current_pool, ref_pool, self.domain_embed(domain_ids)], 1))
    weights, indices = torch.topk(probs, self.top_k, dim=1)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    mixed = torch.zeros_like(current_feat)
    for rank, index in enumerate(indices[0].tolist()):
        value = self.experts[index](current_feat, ref_feat, domain_ids)
        mixed[0] += weights[0, rank].view(1, 1, 1) * value[0]
    zero = current_feat.new_zeros(())
    return current_feat + 0.5 * mixed, zero, zero


def enable_sparse_inference(model):
    """Keep all stored experts, execute only the selected pair at inference."""
    if model.training:
        raise ValueError('Call eval() before enabling sparse inference')
    moe = model.detector.neck.mixture_of_experts
    if (moe.num_experts, moe.top_k, moe.topk_strategy) != (6, 2, 'soft'):
        raise ValueError('Expected six experts and soft Top-2')
    moe.forward = MethodType(_sparse_forward, moe)
    return model


def decode(outputs, original_size, input_size, settings):
    """Return original-image xywh boxes; no GT is used in decoding or NMS."""
    candidates = []
    for value in outputs:
        if value.ndim != 4 or value.shape[0:2] != (1, 6) or not torch.isfinite(value).all():
            raise ValueError('Expected finite batch-one single-class output')
        height, width = value.shape[-2:]
        if height != width:
            raise ValueError('Expected square feature maps')
        raw = value.float().flatten(2).transpose(1, 2)[0]
        yy, xx = torch.meshgrid(torch.arange(height, device=value.device),
                                torch.arange(width, device=value.device), indexing='ij')
        grid = torch.stack((xx, yy), -1).reshape(-1, 2)
        scores = raw[:, 4].sigmoid() * raw[:, 5].sigmoid()
        keep = scores >= settings['score_floor']
        raw, grid, scores = raw[keep], grid[keep], scores[keep]
        stride = input_size / height
        center = (raw[:, :2] + grid) * stride
        extent = raw[:, 2:4].exp() * stride
        boxes = torch.cat((center - extent / 2, center + extent / 2), -1)
        if not torch.isfinite(boxes).all():
            raise ValueError('Nonfinite decoded box above the confidence floor')
        candidates.append((boxes, scores))
    boxes = torch.cat([item[0] for item in candidates])
    scores = torch.cat([item[1] for item in candidates])
    selected = nms(boxes, scores, settings['nms_iou'])[:settings['max_detections']]
    boxes, scores = boxes[selected], scores[selected]
    original_width, original_height = original_size
    # Preserve division order in the archived stretch decoder.
    boxes[:, [0, 2]] /= input_size / original_width
    boxes[:, [1, 3]] /= input_size / original_height
    boxes[:, 2:4] -= boxes[:, :2]
    return [{'bbox': box, 'score': score, 'category_id': 1}
            for box, score in zip(boxes.cpu().tolist(), scores.cpu().tolist())]


class Predictor:
    def __init__(self, config, weights, device='cpu', *, native=False, verify_weights=False):
        seed_everything(config['seed'])
        self.config, self.device = config, torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; use --device cpu')
        self.model = build_model(config)
        load_weights(self.model, weights, config['checkpoint']['sha256'] if verify_weights else None)
        self.model.to(self.device).eval()
        if not native:
            enable_sparse_inference(self.model)

    @torch.inference_mode()
    def __call__(self, path, domain):
        if domain not in self.config['domain_ids'].values():
            raise ValueError('Known domain ID must be 1, 2 or 3')
        size = self.config['prediction']['input_size']
        clip, dimensions = inference_clip(path, image_size=size, frames=self.config['model']['frames'])
        output, _ = self.model(clip.to(self.device), torch.tensor([domain], device=self.device))
        return decode(output, dimensions, size, self.config['scoring']), dimensions
