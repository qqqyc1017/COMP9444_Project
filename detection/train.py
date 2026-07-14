from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO11 insect detector")
    parser.add_argument("--data", required=True, help="YOLO data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained YOLO11 weights")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, 0, 0,1, ...")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="yolo11_pest")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def auto_device(requested: str):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    data = Path(args.data)
    if not data.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {data}")
    model = YOLO(args.model)
    results = model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, device=auto_device(args.device), project=args.project,
        name=args.name, patience=args.patience, seed=args.seed,
        deterministic=True, resume=args.resume,
    )
    print(f"Run directory: {results.save_dir}")
    print(f"Best weights: {Path(results.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()

