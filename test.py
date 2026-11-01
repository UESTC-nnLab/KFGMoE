"""Evaluate KFGMoE or score an existing consolidated prediction file."""
import argparse
import gzip
import json
from pathlib import Path
import time
from kfgmoe.common import load_config, sha256, write_json
from kfgmoe.data import read_annotations
from kfgmoe.inference import Predictor
from kfgmoe.metrics import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--predictions', type=Path, help='Score a saved predictions.jsonl.gz without model inference')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--native', action='store_true', help='Use dense upstream expert execution')
    parser.add_argument('--verify-weights', action='store_true', help='Check the published epoch-60 SHA256')
    args = parser.parse_args()
    config = load_config(args.config)
    if not args.predictions and not all((args.weights, args.annotations, args.data_root)):
        parser.error('Inference requires --weights, --annotations and --data-root')
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    start = time.perf_counter()
    if args.predictions:
        opener = gzip.open if args.predictions.suffix == '.gz' else open
        with opener(args.predictions, 'rt', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        provenance = {'prediction_sha256': sha256(args.predictions)}
    else:
        records = read_annotations(args.annotations, args.data_root)
        model = Predictor(config, args.weights, args.device,
                          native=args.native, verify_weights=args.verify_weights)
        names = {value: name for name, value in config['domain_ids'].items()}
        rows = []
        args.output.mkdir(parents=True, exist_ok=False)
        with gzip.open(args.output / 'predictions.jsonl.gz', 'wt', encoding='utf-8') as stream:
            for index, record in enumerate(records, 1):
                predictions, (width, height) = model(record['path'], record['domain'])
                row = {'image_id': index, 'key': record['key'], 'domain': names[record['domain']],
                       'width': width, 'height': height, 'gt_boxes': record['boxes'][:, :4].tolist(),
                       'predictions': predictions}
                stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
                rows.append(row)
                if index % 100 == 0:
                    print(f'{index}/{len(records)} frames', flush=True)
        provenance = {'checkpoint_sha256': sha256(args.weights),
                      'annotations_sha256': sha256(args.annotations), 'config': config,
                      'execution': 'native' if args.native else 'sparse_top2',
                      'prediction_sha256': sha256(args.output / 'predictions.jsonl.gz')}
    result = evaluate(rows, score_threshold=config['scoring']['operating_threshold'])
    args.output.mkdir(parents=True, exist_ok=True)
    result['provenance'] = provenance
    result['elapsed_seconds'] = time.perf_counter() - start
    write_json(args.output / 'metrics.json', result)
    for domain, metrics in list(result['per_domain'].items()) + [('Macro', result['macro'])]:
        print(domain, {k: None if metrics[k] is None else round(metrics[k] * (1 if k == 'Fa' else 100), 4)
                       for k in ('mAP50', 'mAP75', 'mAP50_95', 'Pr', 'Rc', 'F1', 'Fa')})


if __name__ == '__main__':
    main()
