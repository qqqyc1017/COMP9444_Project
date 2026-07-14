from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11 image, folder, video or webcam detection")
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--source", required=True, help="Image, directory, video, URL, or webcam index")
    parser.add_argument("--output-dir", default="outputs/detection")
    parser.add_argument("--name", default="predict")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--show", action="store_true", help="Open a display window")
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.model).exists():
        raise FileNotFoundError(args.model)
    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.model)
    kwargs = dict(
        source=source, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        save=True, save_txt=True, save_conf=True, stream=True, show=args.show,
        project=args.output_dir, name=args.name, exist_ok=True,
    )
    if args.device is not None:
        kwargs["device"] = args.device
    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "detections.csv"
    fields = ["source", "frame", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for frame_index, result in enumerate(model.predict(**kwargs)):
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                writer.writerow({
                    "source": str(result.path), "frame": frame_index,
                    "class_id": class_id, "class_name": result.names[class_id],
                    "confidence": f"{float(box.conf.item()):.6f}",
                    "x1": f"{x1:.2f}", "y1": f"{y1:.2f}",
                    "x2": f"{x2:.2f}", "y2": f"{y2:.2f}",
                })
    print(f"Annotated media and labels: {output_dir}")
    print(f"Detection table: {csv_path}")


if __name__ == "__main__":
    main()
