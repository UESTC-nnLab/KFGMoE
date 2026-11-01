"""Convert upstream path/xyxy0 annotation text to portable KFGMoE JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kfgmoe.data import read_annotations


def convert(input_path, output_path, data_root, strip_prefix=None, domain=None):
    root = Path(data_root).resolve()
    rows = []
    prefix = None if strip_prefix is None else strip_prefix.replace("\\", "/").rstrip("/") + "/"
    aliases = {"DAUB-H": 1, "IRDST": 2, "IRDST-R": 2, "ITSDT": 3, "ITSDT-15K": 3}
    for number, line in enumerate(Path(input_path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        source = parts[0].replace("\\", "/")
        if prefix:
            if not source.startswith(prefix):
                raise ValueError(f"Line {number}: image path does not start with --strip-prefix")
            relative = source[len(prefix):]
        elif Path(source).is_absolute():
            relative = Path(source).resolve().relative_to(root).as_posix()
        else:
            relative = source
        candidates = {aliases[part] for part in PurePosixPath(relative).parts if part in aliases}
        identifier = domain
        if identifier is None:
            if len(candidates) != 1:
                raise ValueError(f"Line {number}: cannot infer domain; specify --domain")
            identifier = candidates.pop()
        boxes = [list(map(float, value.split(","))) for value in parts[1:]]
        rows.append({"path": relative, "domain": identifier, "boxes": boxes})
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate all rows through the same reader before publishing the output.
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        read_annotations(temporary, root)
        temporary.rename(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--strip-prefix", help="Remove this old machine's image root from input paths")
    parser.add_argument("--domain", type=int, choices=(1, 2, 3), help="Override path-based domain inference")
    args = parser.parse_args()
    count = convert(args.input, args.output, args.data_root, args.strip_prefix, args.domain)
    print(json.dumps({"annotations": count, "output": str(args.output)}))


if __name__ == "__main__":
    main()
