from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a YOLO11 insect detector")
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--data", required=True, help="YOLO data.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="evaluation")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def mean_or_none(value):
    try:
        return float(value.mean())
    except (AttributeError, TypeError, ValueError):
        return None


def main():
    args = parse_args()
    for path in (Path(args.model), Path(args.data)):
        if not path.exists():
            raise FileNotFoundError(path)
    model = YOLO(args.model)
    kwargs = dict(
        data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
        project=args.project, name=f"{args.name}_{args.split}", plots=True,
    )
    if args.device is not None:
        kwargs["device"] = args.device
    metrics = model.val(**kwargs)
    summary = {
        "model": str(Path(args.model).resolve()), "split": args.split,
        "mAP50_95": float(metrics.box.map), "mAP50": float(metrics.box.map50),
        "mAP75": float(metrics.box.map75), "precision": mean_or_none(metrics.box.p),
        "recall": mean_or_none(metrics.box.r), "save_dir": str(metrics.save_dir),
    }
    output = Path(args.output_json) if args.output_json else Path(metrics.save_dir) / "metrics_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

