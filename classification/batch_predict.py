from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import torch
from tqdm import tqdm

from inference_utils import classify_tensor_batch, load_classifier


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Batch classify insect images")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", default="batch_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.0,
                        help="Below this Top-1 confidence, prediction is marked uncertain")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    model, classes, transform, device = load_classifier(args.checkpoint, args.device)
    root = Path(args.input_dir)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No supported images found under {root}")
    output = Path(args.output_csv); output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "status"]
    for rank in range(1, min(args.top_k, len(classes)) + 1):
        fieldnames.extend([f"top{rank}_class", f"top{rank}_confidence"])
    rows, tensors, valid_paths = [], [], []

    def flush():
        if not tensors:
            return
        predictions = classify_tensor_batch(model, torch.stack(tensors), classes, device, args.top_k)
        for path, prediction in zip(valid_paths, predictions):
            row = {"image": str(path), "status": "ok" if prediction[0]["confidence"] >= args.confidence else "uncertain"}
            for rank, item in enumerate(prediction, 1):
                row[f"top{rank}_class"] = item["class"]
                row[f"top{rank}_confidence"] = f'{item["confidence"]:.6f}'
            rows.append(row)
        tensors.clear(); valid_paths.clear()

    for path in tqdm(paths, desc="Images"):
        try:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
            valid_paths.append(path)
        except (OSError, UnidentifiedImageError) as error:
            rows.append({"image": str(path), "status": f"error: {error}"})
        if len(tensors) >= args.batch_size:
            flush()
    flush()
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    print(f"Processed {len(paths)} images; results saved to {output}")


if __name__ == "__main__":
    main()

