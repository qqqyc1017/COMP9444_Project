from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image
import torch
from tqdm import tqdm

from inference_utils import classify_tensor_batch, load_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Classify insects in a video")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-video", default="annotated_video.mp4")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Run inference every N frames and reuse the last result between them")
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1")
    model, classes, transform, device = load_classifier(args.checkpoint, args.device)
    capture = cv2.VideoCapture(str(Path(args.input_video)))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.input_video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output = Path(args.output_video); output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create output video: {output}")
    last_label, last_confidence = "waiting", 0.0
    try:
        for frame_index in tqdm(range(frame_count), desc="Frames"):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % args.frame_step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = transform(Image.fromarray(rgb)).unsqueeze(0)
                prediction = classify_tensor_batch(model, tensor, classes, device, top_k=1)[0][0]
                last_label, last_confidence = prediction["class"], prediction["confidence"]
            shown_label = last_label if last_confidence >= args.confidence else "uncertain"
            text = f"{shown_label}  {last_confidence:.1%}"
            font_scale = max(0.6, min(width, height) / 900)
            thickness = max(1, round(font_scale * 2))
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(frame, (12, 12), (28 + text_w, 34 + text_h), (0, 0, 0), -1)
            color = (60, 220, 60) if last_confidence >= args.confidence else (0, 180, 255)
            cv2.putText(frame, text, (20, 25 + text_h), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness, cv2.LINE_AA)
            writer.write(frame)
    finally:
        capture.release(); writer.release()
    print(f"Annotated video saved to {output}")


if __name__ == "__main__":
    main()
