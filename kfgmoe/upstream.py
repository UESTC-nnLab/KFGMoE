"""Load the exact external CoMoE files used by the archived checkpoint."""
from pathlib import Path
import importlib
import sys
from .common import ROOT, sha256


def import_upstream(config):
    directory = (ROOT / config['upstream']['root']).resolve()
    for name, expected in config['upstream']['files'].items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError('Run python tools/prepare_upstream.py first; missing ' + name)
        if sha256(path) != expected:
            raise ValueError('CoMoE source does not match the pinned version: ' + name)
    for name in ('nets', 'utils'):
        loaded = sys.modules.get(name)
        if loaded is not None and (not getattr(loaded, '__file__', None)
                or not Path(loaded.__file__).resolve().is_relative_to(directory)):
            raise RuntimeError('Another nets/utils package is loaded; use a fresh process')
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    return importlib.import_module('nets.slowfastnet')
