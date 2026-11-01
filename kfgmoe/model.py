"""Checkpoint-compatible KFGMoE model and supervised training objective.

The module names detector.* and evidence.* preserve the released EMA format.
The CoMoE backbone is loaded from the pinned external dependency.
"""
from __future__ import annotations
from dataclasses import dataclass
import importlib
import torch
from torch import nn
import torch.nn.functional as F
from .common import sha256
from .upstream import import_upstream

@dataclass
class PaperOutput:
    detections: list[torch.Tensor]
    native_losses: dict[str, torch.Tensor]
    response_logits: torch.Tensor | None
    input_hw: tuple[int, int]
    gate_probabilities: tuple[torch.Tensor, ...]
    residual_logits: torch.Tensor | None
    base_keyframe_logits: torch.Tensor | None = None


def box_response_mask(targets, logits, input_hw):
    """Coarse box occupancy, not segmentation: every intersecting cell is positive.

    Targets follow the existing detector's letterboxed (cx,cy,w,h,class=0)
    contract. Positive sub-cell boxes survive; outside/degenerate boxes contribute
    no cells. Only KEYFRAME boxes may be passed here.
    """
    if logits.ndim != 4 or logits.shape[1] != 1 or len(targets) != logits.shape[0]:
        raise ValueError("Response/target batch shape mismatch")
    if len(input_hw) != 2 or min(input_hw) <= 0:
        raise ValueError("Invalid input geometry")
    _, _, height, width = logits.shape
    iy, ix = torch.meshgrid(torch.arange(height, device=logits.device),
                            torch.arange(width, device=logits.device), indexing="ij")
    masks = []
    for target in targets:
        boxes = torch.as_tensor(target, device=logits.device, dtype=torch.float32)
        if boxes.ndim != 2 or boxes.shape[1] != 5 or not torch.isfinite(boxes).all():
            raise ValueError("Expected finite Nx5 cxcywh targets")
        if (boxes[:, 4] != 0).any() or (boxes[:, 2:4] < 0).any():
            raise ValueError("Invalid class or negative box extent")
        lo, hi = boxes[:, :2] - boxes[:, 2:4] / 2, boxes[:, :2] + boxes[:, 2:4] / 2
        scale = boxes.new_tensor([width / input_hw[1], height / input_hw[0]])
        lo, hi = lo * scale, hi * scale
        occupied = ((ix[None] < hi[:, 0, None, None]) & (ix[None] + 1 > lo[:, 0, None, None])
                    & (iy[None] < hi[:, 1, None, None]) & (iy[None] + 1 > lo[:, 1, None, None])
                    & (boxes[:, 2, None, None] > 0) & (boxes[:, 3, None, None] > 0))
        masks.append(occupied.any(dim=0))
    return torch.stack(masks).unsqueeze(1).to(dtype=torch.float32)


def balanced_response_loss(logits, mask):
    """Average positive/negative BCE per image, then average present groups.

    An empty-target image has only the negative group; a full mask has only the
    positive group. Both remain finite and trainable without arbitrary pos_weight.
    """
    if logits.shape != mask.shape:
        raise ValueError("Response mask geometry mismatch")
    loss = F.binary_cross_entropy_with_logits(logits.float(), mask.float(), reduction="none")
    positive, negative = mask.flatten(1), (1 - mask).flatten(1)
    values = loss.flatten(1)
    positive_count, negative_count = positive.sum(1), negative.sum(1)
    means = ((values * positive).sum(1) / positive_count.clamp_min(1)
             + (values * negative).sum(1) / negative_count.clamp_min(1))
    groups = (positive_count > 0).int() + (negative_count > 0).int()
    return (means / groups.clamp_min(1)).mean()


class TargetEvidence(nn.Module):
    def __init__(self, channels, experts, settings, *, residual):
        super().__init__()
        self.epsilon = settings["pool_epsilon"]
        self.bound = settings["residual_bound"]
        self.response = nn.Sequential(nn.Conv2d(channels, settings["response_hidden"], 1), nn.GELU(),
                                      nn.Conv2d(settings["response_hidden"], 1, 1))
        self.residual = None
        if residual:
            self.residual = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, settings["routing_hidden"]),
                                          nn.GELU(), nn.Linear(settings["routing_hidden"], experts))
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features):
        logits = self.response(features)
        if self.residual is None:
            return logits, None
        # FP32 reductions prevent overflow/underflow under an external AMP caller.
        weights, values = logits.float().sigmoid(), features.float()
        foreground = (weights * values).sum((2, 3)) / (weights.sum((2, 3)) + self.epsilon)
        descriptor = foreground - values.mean((2, 3))
        delta = self.bound * self.residual(descriptor).float().tanh()
        return logits, delta


class _KeyframeModel(nn.Module):
    def __init__(self, detector, variant, settings):
        super().__init__()
        if variant not in ("B0", "Baux", "M"):
            raise ValueError("Unknown variant")
        self.detector, self.variant = detector, variant
        self.auxiliary_weight = settings["auxiliary_weight"]
        self.evidence = None if variant == "B0" else TargetEvidence(
            detector.neck.mixture_of_experts.channels,
            detector.neck.mixture_of_experts.num_experts, settings, residual=variant == "M")
        self._in_forward = False

    def forward(self, images, domain_ids, *, routing_enabled=True):
        # No ground-truth argument exists here, in train OR eval.
        if images.ndim != 5 or images.shape[1:3] != (3, self.detector.num_frame):
            raise ValueError("Expected B,3,5,H,W causal input")
        if (images.shape[-1] != images.shape[-2] or images.shape[-1] < 64 or images.shape[-1] % 32
                or domain_ids.shape != (images.shape[0],) or domain_ids.dtype != torch.long
                or ((domain_ids < 1) | (domain_ids > 3)).any()):
            raise ValueError("Invalid input geometry or known-domain IDs")
        if self._in_forward:
            raise RuntimeError("Concurrent/reentrant forward on the same model is unsupported")
        self._in_forward = True
        moe = self.detector.neck.mixture_of_experts
        state = {"calls": 0, "keyframe": False, "response": None, "delta": None, "losses": None, "base_logits": None}
        gates, handles = [], []

        def before_moe(_module, args):
            state["calls"] += 1
            keyframe = len(args) > 1 and args[1] is None
            if keyframe != (state["calls"] == self.detector.num_frame):
                raise RuntimeError("Upstream keyframe/history MoE call contract changed")
            state["keyframe"] = keyframe
            if keyframe and self.evidence is not None:
                state["response"], state["delta"] = self.evidence(args[0])

        def modify_logits(_module, _args, result):
            if state["keyframe"]:
                state["base_logits"] = result.detach()
            if state["keyframe"] and self.variant == "M" and routing_enabled:
                return result + state["delta"].to(result.dtype)
            return result

        def capture_gates(_module, _args, result):
            gates.append(result.detach())

        def capture_losses(_module, _args, result):
            state["losses"] = result[1]

        try:
            if not isinstance(moe.gate_network[-1], nn.Softmax) or not isinstance(moe.gate_network[-2], nn.Linear):
                raise RuntimeError("Upstream gate layout changed")
            handles.append(moe.register_forward_pre_hook(before_moe))
            handles.append(moe.gate_network[-2].register_forward_hook(modify_logits))
            handles.append(moe.gate_network[-1].register_forward_hook(capture_gates))
            handles.append(self.detector.neck.register_forward_hook(capture_losses))
            detections, _discarded_item_loss = self.detector(images, domain_ids)
            if state["calls"] != self.detector.num_frame or len(gates) != self.detector.num_frame or state["losses"] is None:
                raise RuntimeError("Upstream neck call/auxiliary loss contract changed")
            return PaperOutput(detections, state["losses"], state["response"], tuple(images.shape[-2:]),
                               tuple(gates), None if state["delta"] is None else state["delta"].detach(),
                               state["base_logits"])
        finally:
            for handle in handles:
                handle.remove()
            self._in_forward = False


OMITTED = {
    'neck.hyperspherical_prototype.temperature', 'neck.adaptive_noise.noise_scale',
    'neck.final_enhance.gamma', 'neck.loss_weights.consistency',
    'neck.loss_weights.diversity_loss', 'neck.loss_weights.load_balance',
    'neck.loss_weights.prototype',
}


class NativeOutputs(list):
    """Carry predicted response to the loss, never targets into routing."""
    def __init__(self, result):
        super().__init__(result.detections)
        self.response_logits = result.response_logits
        self.input_hw = result.input_hw
        self.gate_probabilities = result.gate_probabilities
        self.residual_logits = result.residual_logits
        self.base_keyframe_logits = result.base_keyframe_logits


class KFGMoE(_KeyframeModel):
    def forward(self, images, domain_ids, *, routing_enabled=True):
        result = super().forward(images, domain_ids, routing_enabled=routing_enabled)
        # Same Python accumulation/order as slowfastnet.forward, not tensor.sum().
        native = 0
        for value in result.native_losses.values():
            native += value.item()
        return NativeOutputs(result), native


class ResponseCriterion(nn.Module):
    def __init__(self, native, weight):
        super().__init__()
        self.native, self.weight = native, weight

    def forward(self, outputs, targets):
        detection = self.native(outputs, targets)
        if outputs.response_logits is None:
            return detection
        # The public loader represents a truly empty annotation as shape (0,),
        # while its YOLO loss accepts it by len(). Canonicalize only this empty
        # representation for the response mask; keep native detection targets intact.
        mask_targets = [target.reshape(0, 5) if tuple(target.shape) == (0,) else target
                        for target in targets]
        mask = box_response_mask(mask_targets, outputs.response_logits, outputs.input_hw)
        return detection + self.weight * balanced_response_loss(outputs.response_logits, mask)


def _release_usage(module, inputs, output):
    # Upstream auxiliary losses are converted to scalars; their saved graph is unused.
    moe = module.neck.mixture_of_experts
    moe.expert_usage = moe.expert_usage.detach()


def build_model(config, initial_weights=None):
    source = import_upstream(config)
    original = source.OptimizedHyperMoENeck
    def make_neck(*args, **kwargs):
        kwargs.update(num_experts=config["model"]["experts"],
                      num_frame=config["model"]["frames"], top_k=config["model"]["top_k"])
        return original(*args, **kwargs)
    source.OptimizedHyperMoENeck = make_neck
    try:
        detector = source.slowfastnet(num_classes=1, num_frame=config["model"]["frames"])
    finally:
        source.OptimizedHyperMoENeck = original
    detector.register_forward_hook(_release_usage)
    if initial_weights is not None:
        if sha256(initial_weights) != config["initialization"]["sha256"]:
            raise ValueError("Pretraining weight SHA256 does not match the recorded initialization")
        state = torch.load(initial_weights, map_location="cpu", weights_only=True)
        importlib.import_module("nets.yolo_training").weights_init(detector)
        target = detector.state_dict()
        loaded = {k: v for k, v in state.items() if k in target and target[k].shape == v.shape}
        if not any(k.startswith("backbone.") for k in loaded):
            raise ValueError("No matching pretrained backbone parameters")
        target.update(loaded)
        detector.load_state_dict(target, strict=True)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"])
        model = KFGMoE(detector, "M", config["method"])
    return model


def load_weights(model, path, expected_sha=None):
    if expected_sha is not None and sha256(path) != expected_sha:
        raise ValueError("Checkpoint SHA256 mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model

def optimizer_for(model, config):
    """Exact public module-based grouping, also applied to the new branch."""
    pg0, pg1, pg2 = [], [], []
    for name, module in model.named_modules():
        if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
            pg2.append(module.bias)
        if isinstance(module, nn.BatchNorm2d) or 'bn' in name:
            pg0.append(module.weight)
        elif hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
            pg1.append(module.weight)
    settings = config['training']
    optimizer = torch.optim.SGD(pg0, settings['initial_lr_fit'],
                                momentum=settings['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': settings['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    included = [id(p) for group in optimizer.param_groups for p in group['params']]
    missing = {n.removeprefix('detector.') for n, p in model.named_parameters()
               if id(p) not in set(included)}
    if len(included) != len(set(included)) or missing != OMITTED:
        raise ValueError(f'Native optimizer coverage changed: {missing}')
    return optimizer


def criterion_for(config):
    native = importlib.import_module('nets.yolo_training').YOLOLoss(1, False, strides=[8])
    return ResponseCriterion(native, config['method']['auxiliary_weight'])
