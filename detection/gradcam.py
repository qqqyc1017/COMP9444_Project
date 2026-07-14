from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "classification"))
from inference_utils import load_classifier  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Explain classifier decisions inside YOLO detections")
    parser.add_argument("--detector", required=True, help="YOLO best.pt")
    parser.add_argument("--classifier", required=True, help="ResNet18/ConvNeXt best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/gradcam.jpg")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--top-k-boxes", type=int, default=5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def target_layer(model, name):
    if name == "resnet18":
        return model.layer4[-1]
    if name == "convnext_tiny":
        return model.features[-1]
    raise ValueError(f"Unsupported classifier: {name}")


def main():
    args = parse_args()
    for path in (args.detector, args.classifier, args.image):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    detector = YOLO(args.detector)
    classifier, classes, transform, device = load_classifier(args.classifier, args.device)
    classifier_name = __import__("torch").load(args.classifier, map_location="cpu", weights_only=False)["args"]["model"]
    cam = GradCAM(model=classifier, target_layers=[target_layer(classifier, classifier_name)])
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Cannot read image: {args.image}")
    result = detector.predict(image, conf=args.conf, verbose=False)[0]
    boxes = sorted(result.boxes, key=lambda box: float(box.conf.item()), reverse=True)[:args.top_k_boxes]
    output_image = image.copy(); height, width = image.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        roi_bgr = image[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(roi_rgb)).unsqueeze(0).to(device)
        logits = classifier(tensor); class_id = int(logits.argmax(1).item())
        grayscale = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(class_id)])[0]
        base = cv2.resize(roi_rgb.astype(np.float32) / 255.0, (grayscale.shape[1], grayscale.shape[0]))
        overlay = show_cam_on_image(base, grayscale, use_rgb=True)
        overlay = cv2.resize(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), (x2 - x1, y2 - y1))
        output_image[y1:y2, x1:x2] = overlay
        label = f"{classes[class_id]} | det {float(box.conf.item()):.2f}"
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_image, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), output_image)
    print(f"Grad-CAM visualization saved to {output}")
    print("Note: this explains the classification decision within each YOLO crop, not the YOLO detector itself.")


if __name__ == "__main__":
    main()

