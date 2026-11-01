"""Run detection on one numeric-frame video and save a consolidated JSONL file."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
from kfgmoe.common import load_config
from kfgmoe.inference import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--sequence', type=Path, required=True, help='Directory of frames from ONE video')
    parser.add_argument('--domain', choices=('DAUB-H', 'IRDST-R', 'ITSDT-15K'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--verify-weights', action='store_true')
    parser.add_argument('--visualize', type=int, default=8, help='Maximum number of preview images (default: 8)')
    parser.add_argument('--confidence', type=float, default=.5, help='Preview threshold; JSON preserves the evaluation floor')
    args = parser.parse_args()
    if args.visualize < 0 or not 0 <= args.confidence <= 1:
        parser.error('Invalid preview limit or confidence')
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    frames = sorted((p for p in args.sequence.iterdir() if p.suffix.lower() == '.bmp' and p.stem.isdigit()),
                    key=lambda p: int(p.stem))
    if not frames:
        parser.error('Sequence must contain numeric BMP frames')
    config = load_config(args.config)
    predictor = Predictor(config, args.weights, args.device, native=args.native, verify_weights=args.verify_weights)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'detections.jsonl').open('x', encoding='utf-8') as stream:
        for index, path in enumerate(frames):
            detections, (width, height) = predictor(path, config['domain_ids'][args.domain])
            row = {'frame': path.name, 'width': width, 'height': height, 'predictions': detections}
            stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
            if index < args.visualize:
                with Image.open(path) as source:
                    canvas = source.convert('RGB')
                draw = ImageDraw.Draw(canvas)
                for detection in detections:
                    if detection['score'] >= args.confidence:
                        x, y, w, h = detection['bbox']
                        draw.rectangle((x, y, x+w, y+h), outline=(0, 230, 100), width=2)
                        draw.text((x, max(0, y-12)), f"{detection['score']:.2f}", fill=(0, 230, 100))
                canvas.save(args.output / f'preview_{index:03d}.png')
            if (index + 1) % 100 == 0:
                print(f'{index+1}/{len(frames)} frames', flush=True)
    print(f'Saved {len(frames)} frames to {args.output}')


if __name__ == '__main__':
    main()
