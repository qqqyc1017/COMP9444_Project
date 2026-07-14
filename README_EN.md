# Complete IP102 Insect Pest Recognition Project

This project provides two complementary workflows:

1. **Image classification**: train and compare a ResNet18 baseline and ConvNeXt-Tiny on IP102.
2. **Video object detection**: use an existing YOLO11 `best.pt` to locate and recognize multiple insects in a video.

## 1. Project structure

```text
insect_pest_project_final/
├── classification/
│   ├── train.py
│   ├── models.py
│   ├── data.py
│   ├── engine.py
│   ├── compare.py
│   ├── inference_utils.py
│   ├── predict.py
│   ├── batch_predict.py
│   └── video_predict.py
├── yolo_video/
│   └── detect_video.py
├── requirements.txt
├── README_ZH.md
└── README_EN.md
```

## 2. Environment

Python 3.10 or 3.11 is recommended:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 3. Download IP102 classification data

Official repository:

```text
https://github.com/xpwu95/IP102
```

Official Google Drive:

```text
https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo?usp=sharing
```

Open `IP102_v1.1/Classification` and download the classification archive. After extraction, point `--data-dir` to the directory containing `train.txt`, `val.txt`, `test.txt`, and the image directory.

## 4. Classification files

- `train.py`: trains ResNet18 or ConvNeXt-Tiny and saves the best checkpoint and test metrics.
- `models.py`: creates both models and replaces their output heads.
- `data.py`: reads official split files or ImageFolder data and applies augmentation.
- `engine.py`: shared train/evaluate loop, metrics, latency, curves, and confusion matrices.
- `compare.py`: generates the comparison CSV and chart.
- `inference_utils.py`: shared checkpoint loading and preprocessing.
- `predict.py`: Top-K prediction for one image.
- `batch_predict.py`: batch image prediction with CSV output.
- `video_predict.py`: whole-frame video classification without bounding boxes.

## 5. Train and compare classifiers

Run a one-epoch smoke test first:

```bash
python classification/train.py --data-dir /path/to/IP102 --model resnet18 --epochs 1 --batch-size 8 --workers 0 --output-dir outputs/smoke_test
```

Formal experiments:

```bash
python classification/train.py --data-dir /path/to/IP102 --model resnet18 --epochs 30 --batch-size 64 --seed 42 --output-dir outputs/classification
python classification/train.py --data-dir /path/to/IP102 --model convnext_tiny --epochs 30 --batch-size 32 --seed 42 --output-dir outputs/classification
python classification/compare.py --output-dir outputs/classification
```

Both models must use the same splits, image size, augmentation, number of epochs, random seed, and test set.

## 6. Classification inference

Single image:

```bash
python classification/predict.py --checkpoint outputs/classification/convnext_tiny/best.pt --image insect.jpg --top-k 5
```

Image directory:

```bash
python classification/batch_predict.py --checkpoint outputs/classification/convnext_tiny/best.pt --input-dir images --output-csv outputs/classification/batch.csv
```

Whole-frame video classification:

```bash
python classification/video_predict.py --checkpoint outputs/classification/convnext_tiny/best.pt --input-video input.mp4 --output-video outputs/classification/video.mp4
```

## 7. Detect video with an existing YOLO11 best.pt

You need a trained YOLO11 detection checkpoint and an input video. No YOLO retraining or detection dataset is required.

Check the model task and class names:

```bash
python -c "from ultralytics import YOLO; m=YOLO('/path/to/best.pt'); print('task:', m.task); print('classes:', m.names)"
```

The task must be `detect`.

Run video detection:

```bash
python yolo_video/detect_video.py \
  --model /path/to/best.pt \
  --input-video /path/to/input.mp4 \
  --output-video outputs/yolo_video.mp4 \
  --output-csv outputs/yolo_detections.csv \
  --conf 0.25 \
  --iou 0.7 \
  --imgsz 640
```

Use `--device 0` for an NVIDIA GPU, `--device mps` for Apple Silicon, or `--device cpu` for CPU inference. If omitted, Ultralytics selects the device automatically.

The output video contains bounding boxes, class names, and confidence scores. The CSV records the frame, timestamp, class, confidence, and box coordinates for every detection.

OpenCV output normally does not preserve the original audio track. FFmpeg can be used to merge the source audio back afterward.

## 8. Difference between the video scripts

| Script | Model | Function |
|---|---|---|
| `classification/video_predict.py` | ResNet18 or ConvNeXt-Tiny | One main class per frame, no boxes |
| `yolo_video/detect_video.py` | Existing YOLO11 `best.pt` | Multiple detections with bounding boxes |

## 9. Fair experiments and team hand-off

- Use identical train/validation/test splits for both classifiers.
- Record commands, seeds, GPU model, package versions, and training time.
- When resources allow, run seeds 42, 43, and 44 and report mean ± standard deviation.
- Share metrics JSON, history JSON, curves, confusion matrices, and comparison CSV files.
- Share large `.pt` checkpoints through cloud storage rather than Git.

