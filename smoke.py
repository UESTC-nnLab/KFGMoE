"""Check model loading, native/sparse equivalence and a synthetic backward pass."""
import argparse
import copy
import json
from pathlib import Path
import torch
from kfgmoe.common import load_config, seed_everything
from kfgmoe.model import build_model, load_weights, criterion_for, optimizer_for
from kfgmoe.inference import enable_sparse_inference, decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = load_config(args.config)
    seed_everything(config['seed'])
    model = build_model(config)
    if args.weights:
        load_weights(model, args.weights, config['checkpoint']['sha256'])
    model.to(args.device).eval()
    sparse = enable_sparse_inference(copy.deepcopy(model))
    clip = torch.randn(1, 3, 5, 64, 64, device=args.device)
    errors = []
    with torch.no_grad():
        for domain in (1, 2, 3):
            domains = torch.tensor([domain], device=args.device)
            dense_out, _ = model(clip, domains)
            sparse_out, _ = sparse(clip, domains)
            errors.append(max((a-b).abs().max().item() for a, b in zip(dense_out, sparse_out)))
            for a, b in zip(dense_out, sparse_out):
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
            if not torch.equal(dense_out.residual_logits, sparse_out.residual_logits):
                raise AssertionError('Sparse execution changed the keyframe gate correction')
    model.train()
    optimizer = optimizer_for(model, config)
    outputs, native_aux = model(clip.repeat(2, 1, 1, 1, 1), torch.tensor([1, 3], device=args.device))
    targets = [torch.tensor([[32., 32., 8., 8., 0.]], device=args.device), torch.empty(0, 5, device=args.device)]
    loss = criterion_for(config)(outputs, targets) + native_aux
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad for p in model.evidence.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise AssertionError('Missing or nonfinite keyframe-guidance gradients')
    # Exercise empty detections, avoiding exp() on rejected background logits.
    empty = torch.full((1, 6, 8, 8), -100., device=args.device)
    empty[:, 2:4] = 1000.
    assert decode([empty], (83, 77), 64, config['scoring']) == []
    print(json.dumps({'status': 'passed', 'parameters': sum(p.numel() for p in model.parameters()),
                      'native_sparse_max_abs': errors, 'finite_loss': float(loss.detach()),
                      'weights_checked': args.weights is not None}, indent=2))


if __name__ == '__main__':
    main()
