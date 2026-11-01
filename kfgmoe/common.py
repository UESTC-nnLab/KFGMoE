"""Configuration and deterministic execution helpers."""
from pathlib import Path
import hashlib
import json
import os
import random
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_config(path=None):
    path = ROOT / 'configs/kfgmoe.json' if path is None else Path(path)
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    if config['model'] != {'frames': 5, 'experts': 6, 'top_k': 2}:
        raise ValueError('This release targets the five-frame, six-expert, Top-2 checkpoint')
    if config['domain_ids'] != {'DAUB-H': 1, 'IRDST-R': 2, 'ITSDT-15K': 3}:
        raise ValueError('Checkpoint domain IDs must be DAUB-H=1, IRDST-R=2, ITSDT-15K=3')
    train = config['training']
    if not 1 <= train['stop_epoch'] <= train['epochs'] or train['batch_size'] < 1:
        raise ValueError('Invalid epoch or batch configuration')
    for size in (train['input_size'], config['prediction']['input_size']):
        if size < 64 or size % 32:
            raise ValueError('Image size must be a multiple of 32, at least 64')
    return config


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
