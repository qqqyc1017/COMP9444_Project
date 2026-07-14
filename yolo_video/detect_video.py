from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Detect insects in a video using an existing YOLO best.pt")
    parser.add_argument("--model", required=True, help="Path to the trained YOLO best.pt")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-video", default="outputs/yolo_video.mp4")
    parser.add_argument("--output-csv", default="outputs/yolo_detections.csv")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Examples: 0, cpu, mps; omit for automatic selection")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path, input_path = Path(args.model), Path(args.input_video)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")
    model = YOLO(str(model_path))
    if model.task != "detect":
        raise ValueError(f"Expected a YOLO detection model, but best.pt task is {model.task!r}")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video = Path(args.output_video); output_csv = Path(args.output_csv)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create output video: {output_video}")
    fields = ["frame", "time_seconds", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
    try:
        with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=fields); csv_writer.writeheader()
            frame_index = 0
            with tqdm(total=total_frames or None, desc="Video frames") as progress:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    predict_kwargs = {
                        "source": frame, "conf": args.conf, "iou": args.iou,
                        "imgsz": args.imgsz, "verbose": False,
                    }
                    if args.device is not None:
                        predict_kwargs["device"] = args.device
                    result = model.predict(**predict_kwargs)[0]
                    writer.write(result.plot())
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls.item())
                            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                            csv_writer.writerow({
                                "frame": frame_index, "time_seconds": f"{frame_index / fps:.3f}",
                                "class_id": class_id, "class_name": result.names[class_id],
                                "confidence": f"{float(box.conf.item()):.6f}",
                                "x1": f"{x1:.2f}", "y1": f"{y1:.2f}",
                                "x2": f"{x2:.2f}", "y2": f"{y2:.2f}",
                            })
                    frame_index += 1; progress.update(1)
    finally:
        capture.release(); writer.release()
    print(f"Annotated video: {output_video}")
    print(f"Detection CSV: {output_csv}")


if __name__ == "__main__":
    main()

