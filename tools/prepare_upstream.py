"""Prepare the pinned CoMoE dependency from a local checkout or its official source."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Copy from an existing pinned CoMoE directory (offline)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/kfgmoe.json').read_text())['upstream']
    destination = root / config['root']
    prepared = 0
    for name, expected in config['files'].items():
        target = destination / name
        if target.exists():
            payload = target.read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f'Existing source differs; it has not been overwritten: {target}')
            continue
        if args.source:
            payload = (args.source / name).read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/UESTC-nnLab/CoMoE/{config['commit']}/{name}"
            with urlopen(url, timeout=60) as response:
                payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Source SHA256 mismatch: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(payload)
        prepared += 1
    print(f"CoMoE {config['commit']}: {len(config['files'])} files verified ({prepared} prepared).")


if __name__ == '__main__':
    main()
