"""Portable annotation lists and the checkpoint's causal five-frame input geometry."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


DOMAIN_IDS = {"DAUB-H": 1, "IRDST-R": 2, "ITSDT-15K": 3}
MEAN = np.asarray([0.485, 0.456, 0.406])
STD = np.asarray([0.229, 0.224, 0.225])


def read_annotations(annotation_path, data_root):
    """Read JSONL rows: relative ``path``, integer ``domain``, xyxy-class ``boxes``.

    Empty frames use ``boxes: []``. The only class is target (class ID 0).
    Annotation order is retained. No split or sampling is performed here.
    """
    root = Path(data_root).resolve()
    records, seen = [], set()
    with Path(annotation_path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["path"].replace("\\", "/")
            relative = Path(key)
            if relative.is_absolute() or ":" in key or ".." in relative.parts:
                raise ValueError(f"Line {number}: path must be relative to data_root")
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Line {number}: path leaves data_root")
            domain = row["domain"]
            if isinstance(domain, str):
                domain = DOMAIN_IDS.get(domain, domain)
            if type(domain) is not int or domain not in DOMAIN_IDS.values():
                raise ValueError(f"Line {number}: domain must be 1, 2 or 3")
            boxes = np.asarray(row["boxes"], dtype=np.float64)
            if boxes.size == 0:
                boxes = np.empty((0, 5), dtype=np.float64)
            if (boxes.ndim != 2 or boxes.shape[1] != 5 or not np.isfinite(boxes).all()
                    or np.any(boxes[:, 4] != 0) or np.any(boxes[:, 2:4] <= boxes[:, :2])):
                raise ValueError(f"Line {number}: expected finite, positive-area xyxy0 boxes")
            if key in seen:
                raise ValueError(f"Line {number}: repeated image {key}")
            seen.add(key)
            records.append({"path": path, "key": key, "domain": domain, "boxes": boxes})
    if not records:
        raise ValueError("Annotation list is empty")
    return records


def causal_paths(path, frames=5):
    """Return oldest-to-current paths; repeat frame zero at the video boundary."""
    path = Path(path)
    if frames < 1 or not path.stem.isdigit():
        raise ValueError("Expected a positive window and a numeric frame filename")
    current = int(path.stem)
    # Numbered benchmark videos use 0.bmp, 1.bmp, ...; padded names also work.
    padding = len(path.stem) if path.stem.startswith("0") else 0
    result = [path.with_name(str(max(current - lag, 0)).zfill(padding) + path.suffix)
              for lag in reversed(range(frames))]
    for source in result:
        if not source.is_file():
            raise FileNotFoundError(f"Missing causal frame: {source}")
    return result


def normalize(images):
    """Keep the archived float32 NumPy normalization order."""
    value = np.asarray(images, dtype=np.float32)
    value /= 255.0
    value -= MEAN
    value /= STD
    return torch.from_numpy(np.ascontiguousarray(value.transpose(3, 0, 1, 2)))


def inference_clip(path, image_size=512, frames=5):
    """Stretch RGB frames to a square; return B,C,T,H,W and original (W,H)."""
    images, original = [], None
    for source in causal_paths(path, frames):
        with Image.open(source) as image:
            image = image.convert("RGB")
            if original is not None and image.size != original:
                raise ValueError("Frame dimensions must be constant within a clip")
            original = image.size
            images.append(np.asarray(image.resize((image_size, image_size), Image.Resampling.BICUBIC),
                                     dtype=np.float32))
    return normalize(np.stack(images)).unsqueeze(0), original


class CausalDataset(Dataset):
    """Native training letterbox and box quantization, without hidden augmentation.

    The archived training loader does not apply its optional flip routine. Its
    box coordinate assignments use integer arrays; this quantization is retained.
    Labels are copied for each access so repeated calls never alter annotations.
    """

    def __init__(self, records, image_size=512, frames=5, training=True):
        self.records = records
        self.image_size, self.frames, self.training = image_size, frames, training
        if image_size < 64 or image_size % 32 or frames < 1:
            raise ValueError("image_size must be a multiple of 32 and at least 64")
        if training and any(np.any(row["boxes"] != np.floor(row["boxes"])) for row in records):
            raise ValueError("The historical training loader requires integer box coordinates")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        side = self.image_size
        boxes = row["boxes"].astype(np.int64, copy=True)
        images, original = [], None
        # Resize current first to preserve the archived label-shuffle RNG timing.
        for position, source in enumerate(reversed(causal_paths(row["path"], self.frames))):
            with Image.open(source) as image:
                image = image.convert("RGB")
                if original is not None and image.size != original:
                    raise ValueError("Frame dimensions must be constant within a clip")
                original = image.size
                iw, ih = original
                scale = min(side / iw, side / ih)
                nw, nh = int(iw * scale), int(ih * scale)
                dx, dy = (side - nw) // 2, (side - nh) // 2
                canvas = Image.new("RGB", (side, side), (128, 128, 128))
                canvas.paste(image.resize((nw, nh), Image.Resampling.BICUBIC), (dx, dy))
                images.append(np.asarray(canvas, dtype=np.float32))
            if position == 0 and len(boxes):
                if self.training:
                    np.random.shuffle(boxes)
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + dx
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + dy
                boxes[:, :2] = np.maximum(boxes[:, :2], 0)
                boxes[:, 2] = np.minimum(boxes[:, 2], side)
                boxes[:, 3] = np.minimum(boxes[:, 3], side)
                extent = boxes[:, 2:4] - boxes[:, :2]
                boxes = boxes[(extent[:, 0] > 1) & (extent[:, 1] > 1)]
        boxes = boxes.astype(np.float32)
        if len(boxes):
            boxes[:, 2:4] -= boxes[:, :2]
            boxes[:, :2] += boxes[:, 2:4] / 2
        return normalize(np.stack(images[::-1])), torch.from_numpy(boxes), row["domain"]


def collate(batch):
    images, boxes, domains = zip(*batch)
    return torch.stack(images), list(boxes), torch.tensor(domains, dtype=torch.long)


def seed_worker(_worker_id):
    """Derive Python/NumPy worker seeds from the DataLoader's seeded generator."""
    import random
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
