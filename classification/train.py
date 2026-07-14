from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from data import make_datasets, make_loaders
from engine import (benchmark_latency, get_device, run_epoch, save_confusion,
                    save_training_plot, seed_everything, write_json)
from models import MODEL_NAMES, count_parameters, create_model


def parse_args():
    p = argparse.ArgumentParser(description="Train an IP102 insect classifier")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model", choices=MODEL_NAMES, required=True)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-pretrained", action="store_true")
    return p.parse_args()


def main():
    args = parse_args(); seed_everything(args.seed)
    device = get_device(args.device)
    datasets, class_names = make_datasets(args.data_dir, args.image_size)
    loaders = make_loaders(datasets, args.batch_size, args.workers)
    model = create_model(args.model, len(class_names), not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out = Path(args.output_dir) / args.model; out.mkdir(parents=True, exist_ok=True)
    history, best_acc, stale = [], -1.0, 0
    print(f"model={args.model} classes={len(class_names)} device={device}")
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler)
        val_metrics, _, _ = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row); print(row)
        if val_metrics["accuracy"] > best_acc:
            best_acc, stale = val_metrics["accuracy"], 0
            torch.save({"model": model.state_dict(), "classes": class_names, "args": vars(args)}, out / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}"); break
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics, y_true, y_pred = run_epoch(model, loaders["test"], criterion, device)
    result = {"model": args.model, "num_classes": len(class_names), "parameters": count_parameters(model),
              "device": str(device), "best_val_accuracy": best_acc, **{f"test_{k}": v for k, v in test_metrics.items()},
              "latency_ms_batch1": benchmark_latency(model, device, args.image_size)}
    write_json(history, out / "history.json"); write_json(result, out / "metrics.json")
    save_training_plot(history, out / "curves.png")
    save_confusion(y_true, y_pred, class_names, out / "confusion_matrix.png")
    print(result)


if __name__ == "__main__":
    main()

