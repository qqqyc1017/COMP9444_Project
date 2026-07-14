# Complete Insect Pest Recognition Project

[中文说明](README_ZH.md)

This repository contains two complementary tasks:

1. **Image classification** — ResNet18 baseline vs. ConvNeXt-Tiny on IP102.
2. **Object detection** — YOLO11 for locating and classifying one or more insects in images and videos.

Classification and detection are evaluated separately: classification uses Accuracy/Macro-F1; detection uses mAP/Precision/Recall.

## Setup

Use Python 3.10 or 3.11:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Classification: ResNet18 vs. ConvNeXt-Tiny

The classification dataset must contain official `train.txt`, `val.txt`, `test.txt` files plus `images/`, or ImageFolder-style `train/`, `val/`, `test/` directories. Run from the project root:

```bash
python classification/train.py --data-dir /path/to/IP102 --model resnet18 --epochs 30 --output-dir outputs/classification
python classification/train.py --data-dir /path/to/IP102 --model convnext_tiny --epochs 30 --output-dir outputs/classification
python classification/compare.py --output-dir outputs/classification
```

Single image, folder and video classification:

```bash
python classification/predict.py --checkpoint outputs/classification/convnext_tiny/best.pt --image insect.jpg
python classification/batch_predict.py --checkpoint outputs/classification/convnext_tiny/best.pt --input-dir images --output-csv outputs/classification/batch.csv
python classification/video_predict.py --checkpoint outputs/classification/convnext_tiny/best.pt --input-video input.mp4 --output-video outputs/classification/video.mp4
```

## Detection: YOLO11

YOLO requires bounding-box labels; ordinary IP102 classification labels cannot train a detector. Copy `configs/yolo_data.example.yaml`, update `path` and class names, and check that every image has a matching label file.

Train:

```bash
python detection/train.py --data configs/yolo_data.yaml --model yolo11n.pt --epochs 70 --batch 16
```

Evaluate on validation and test sets:

```bash
python detection/evaluate.py --model runs/detect/yolo11_pest/weights/best.pt --data configs/yolo_data.yaml --split val
python detection/evaluate.py --model runs/detect/yolo11_pest/weights/best.pt --data configs/yolo_data.yaml --split test
```

Detect an image or an entire directory:

```bash
python detection/predict.py --model runs/detect/yolo11_pest/weights/best.pt --source images --output-dir outputs/detection --name images
```

Detect a video or webcam stream:

```bash
python detection/predict.py --model runs/detect/yolo11_pest/weights/best.pt --source input.mp4 --output-dir outputs/detection --name video
python detection/predict.py --model runs/detect/yolo11_pest/weights/best.pt --source 0 --output-dir outputs/detection --name webcam --show
```

The prediction directory contains annotated media, YOLO text labels and `detections.csv`.

Feature-attention maps:

```bash
python detection/attention_map.py --model runs/detect/yolo11_pest/weights/best.pt --source images --output-dir outputs/attention
```

Grad-CAM for the trained classification model within YOLO detection boxes:

```bash
python detection/gradcam.py \
  --detector runs/detect/yolo11_pest/weights/best.pt \
  --classifier outputs/classification/convnext_tiny/best.pt \
  --image insect.jpg --output outputs/gradcam.jpg
```

## Fair experiments and team hand-off

- Keep the same train/validation/test splits for both classifiers.
- Run each classifier with seeds 42, 43 and 44 when resources allow; report mean ± standard deviation.
- Record the exact command, GPU, package versions and training time for each run.
- Share JSON/CSV metrics, curves and confusion matrices. Share `.pt` weights separately through cloud storage rather than Git.
- Do not compare classification Accuracy directly with detection mAP; answer different research questions with separate tables.

## Expected outputs

```text
outputs/
├── classification/
│   ├── resnet18/
│   ├── convnext_tiny/
│   ├── comparison.csv
│   └── comparison.png
├── detection/
├── attention/
└── gradcam.jpg
```
