from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate YOLO feature-attention overlays")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True, help="Image or image directory")
    parser.add_argument("--output-dir", default="outputs/attention_maps")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--activation-threshold", type=float, default=0.6)
    return parser.parse_args()


def last_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        tensors = [last_tensor(item) for item in value]
        return next((item for item in reversed(tensors) if item is not None), None)
    return None


def roi_coverage(attention, box, threshold):
    height, width = attention.shape
    x1, y1, x2, y2 = map(int, box)
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = attention[y1:y2, x1:x2]
    return float((roi >= threshold).mean()) if roi.size else 0.0


def image_paths(source: Path):
    if source.is_file():
        return [source]
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in EXTENSIONS)


def main():
    args = parse_args(); source = Path(args.source); output = Path(args.output_dir)
    if not Path(args.model).exists() or not source.exists():
        raise FileNotFoundError("Model or source does not exist")
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model); captured = []

    def hook_fn(_module, _inputs, result):
        tensor = last_tensor(result)
        if tensor is not None:
            captured.append(tensor.detach())

    hook = model.model.model[-2].register_forward_hook(hook_fn)
    rows = []
    try:
        for path in image_paths(source):
            image = cv2.imread(str(path))
            if image is None:
                continue
            captured.clear()
            prediction = model.predict(image, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            if not captured:
                print(f"Warning: no feature tensor captured for {path.name}")
                continue
            feature = captured[-1]
            if feature.ndim != 4:
                print(f"Warning: unsupported feature shape {tuple(feature.shape)}")
                continue
            attention = feature[0].float().mean(0).cpu().numpy()
            spread = float(attention.max() - attention.min())
            attention = np.zeros_like(attention) if spread < 1e-12 else (attention - attention.min()) / spread
            attention = cv2.resize(attention, (image.shape[1], image.shape[0]))
            heatmap = cv2.applyColorMap(np.uint8(attention * 255), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
            scores = [roi_coverage(attention, box.xyxy[0].cpu().numpy(), args.activation_threshold)
                      for box in prediction.boxes]
            mean_score = float(np.mean(scores)) if scores else 0.0
            relative = path.name if source.is_file() else str(path.relative_to(source)).replace("/", "__").replace("\\", "__")
            save_path = output / f"attention_{relative}"
            save_path.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(save_path), overlay)
            rows.append({"image": str(path), "detections": len(scores), "mean_roi_coverage": f"{mean_score:.6f}"})
    finally:
        hook.remove()
    with (output / "roi_scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "detections", "mean_roi_coverage"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Attention maps and ROI scores saved to {output}")


if __name__ == "__main__":
    main()
