"""Train KFGMoE and save EMA weights and a resumable checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kfgmoe.common import load_config, seed_everything
from kfgmoe.data import CausalDataset, collate, read_annotations, seed_worker
from kfgmoe.model import build_model, criterion_for, optimizer_for


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_save(value, destination):
    """Replace only this run's rolling checkpoint after a successful full write."""
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".partial")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)


def write_history(output, history):
    """Restore the small log from checkpoint history after an interrupted save."""
    temporary = output / "training.jsonl.partial"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for entry in history:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
    temporary.replace(output / "training.jsonl")


def capture_rng(generator):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy_name": numpy_state[0],
            "numpy_keys": torch.tensor(numpy_state[1].astype(np.int64)),
            "numpy_position": numpy_state[2], "numpy_has_gauss": numpy_state[3],
            "numpy_cached_gaussian": numpy_state[4], "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state((state["numpy_name"], state["numpy_keys"].numpy().astype(np.uint32),
                         state["numpy_position"], state["numpy_has_gauss"], state["numpy_cached_gaussian"]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/kfgmoe.json"))
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/kfgmoe"))
    parser.add_argument("--resume", type=Path, help="This runner's latest.pt (safe weights-only loading)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, help="Override the number of data workers")
    parser.add_argument("--dry-run", action="store_true", help="Check annotations, one clip and the update budget; no model training")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    cfg = config["training"]
    if args.workers is not None:
        cfg["workers"] = args.workers
    if cfg["workers"] < 0 or not 1 <= cfg["stop_epoch"] <= cfg["epochs"]:
        parser.error("Require workers >= 0 and 1 <= stop_epoch <= epochs")
    if cfg.get("precision", "fp32") != "fp32":
        parser.error("The release training recipe uses FP32")
    seed_everything(config["seed"])
    records = read_annotations(args.train_list, args.data_root)
    dataset = CausalDataset(records, cfg["input_size"], config["model"]["frames"], training=True)
    steps = len(dataset) // cfg["batch_size"] // cfg["native_epoch_divisor"]
    if steps < 1:
        parser.error("Training list is too short for batch_size * native_epoch_divisor")
    if args.dry_run:
        images, boxes, domain = dataset[0]
        print(json.dumps({"frames": len(dataset), "clip_shape": list(images.shape),
                          "first_targets": len(boxes), "first_domain": domain,
                          "steps_per_epoch": steps, "schedule_epochs": cfg["epochs"],
                          "stop_epoch": cfg["stop_epoch"], "updates": steps * cfg["stop_epoch"],
                          "training": False}))
        return
    if not args.initial_weights and not args.resume:
        parser.error("Training requires --initial-weights; evaluation weights are not training initialization")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu only for a small smoke test")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Use a new --output directory or explicitly resume this run")
    if args.resume and args.resume.resolve().parent != args.output.resolve():
        parser.error("--resume must belong to --output")
    identity = {"config": config, "train_list_sha256": digest(args.train_list),
                "steps_per_epoch": steps, "torch_version": str(torch.__version__),
                "initial_weights_sha256": digest(args.initial_weights) if args.initial_weights else None}
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("schema") != "kfgmoe_training_v1":
            raise ValueError("Resume file was not created by this training runner")
        previous = checkpoint["identity"]
        if identity["initial_weights_sha256"] is None:
            identity["initial_weights_sha256"] = previous["initial_weights_sha256"]
        if previous != identity:
            raise ValueError("Configuration, annotation list, initialization or environment changed")
    model = build_model(config, initial_weights=None if checkpoint else args.initial_weights)
    # Match the archived LossHistory dummy draw before loader construction.
    torch.randn(2, 3, 512, 512)
    model = model.to(device).train()
    native = importlib.import_module("nets.yolo_training")
    ema = native.ModelEMA(model)
    optimizer, criterion = optimizer_for(model, config), criterion_for(config).to(device)
    schedule = native.get_lr_scheduler("cos", cfg["initial_lr_fit"], cfg["minimum_lr_fit"], cfg["epochs"])
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["workers"], persistent_workers=False,
                        pin_memory=device.type == "cuda", drop_last=True, collate_fn=collate,
                        worker_init_fn=seed_worker, generator=generator)
    start, history = 0, []
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        ema.ema.load_state_dict(checkpoint["ema"], strict=True)
        ema.updates = checkpoint["ema_updates"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        for name, parameter in model.named_parameters():
            gradient = checkpoint["gradients"].get(name)
            parameter.grad = None if gradient is None else gradient.to(device)
        start = checkpoint["next_epoch"]
        history = checkpoint["history"]
        restore_rng(checkpoint["rng"], generator)
        if start > cfg["stop_epoch"]:
            raise ValueError("The resume checkpoint is beyond the requested stop epoch")
        write_history(args.output, history)
        if start == cfg["stop_epoch"]:
            final = args.output / f"epoch_{start:03d}_ema.pth"
            atomic_save(ema.ema.state_dict(), final)
            print(json.dumps({"completed_epoch": start, "ema_updates": ema.updates,
                              "weights": str(final), "resumed": True}))
            return
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "run.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    for epoch in range(start, cfg["stop_epoch"]):
        native.set_optimizer_lr(optimizer, schedule, epoch)
        model.train()
        total, started = 0.0, time.perf_counter()
        iterator = iter(loader)
        progress = tqdm(range(steps), desc=f"Epoch {epoch + 1}/{cfg['stop_epoch']}")
        for step in progress:
            images, targets, domains = next(iterator)
            images, domains = images.to(device), domains.to(device)
            targets = [target.to(device) for target in targets]
            optimizer.zero_grad()
            outputs, auxiliary = model(images, domains)
            supervised = criterion(outputs, targets)
            loss = supervised + auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}, step {step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
            optimizer.step()
            ema.update(model)
            total += supervised.item()
            progress.set_postfix(loss=total / (step + 1), lr=optimizer.param_groups[0]["lr"])
        del iterator
        entry = {"epoch": epoch + 1, "loss": total / steps, "lr": optimizer.param_groups[0]["lr"],
                 "ema_updates": ema.updates, "seconds": time.perf_counter() - started}
        if not math.isfinite(entry["loss"]):
            raise FloatingPointError("Nonfinite epoch loss")
        history.append(entry)
        snapshot = {"schema": "kfgmoe_training_v1", "identity": identity,
                    "model": model.state_dict(), "ema": ema.ema.state_dict(),
                    "optimizer": optimizer.state_dict(), "ema_updates": ema.updates,
                    "next_epoch": epoch + 1, "rng": capture_rng(generator), "history": history,
                    "gradients": {name: p.grad.detach().cpu() for name, p in model.named_parameters()
                                  if p.grad is not None}}
        atomic_save(snapshot, args.output / "latest.pt")
        write_history(args.output, history)
        if epoch + 1 == 60 or epoch + 1 == cfg["stop_epoch"]:
            atomic_save(ema.ema.state_dict(), args.output / f"epoch_{epoch + 1:03d}_ema.pth")
    print(json.dumps({"completed_epoch": cfg["stop_epoch"], "ema_updates": ema.updates,
                      "weights": str(args.output / f"epoch_{cfg['stop_epoch']:03d}_ema.pth")}))


if __name__ == "__main__":
    main()
