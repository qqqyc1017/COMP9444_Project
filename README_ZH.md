# 昆虫害虫识别完整项目

[English Documentation](README_EN.md)

本项目包含两个相互补充的任务：

1. **图像分类**：在 IP102 数据集上对比 ResNet18 Baseline 与 ConvNeXt-Tiny。
2. **目标检测**：使用 YOLO11 在图片和视频中定位并识别一只或多只昆虫。

分类与检测需要分别评估：分类使用 Accuracy 和 Macro-F1；检测使用 mAP、Precision 和 Recall。两类指标不能直接比较。

## 1. 环境安装

推荐使用 Python 3.10 或 Python 3.11：

```bash
python -m venv .venv
```

macOS/Linux：

```bash
source .venv/bin/activate
```

Windows：

```powershell
.venv\Scripts\activate
```

安装项目依赖：

```bash
pip install -r requirements.txt
```

检查 PyTorch 是否识别 NVIDIA GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 2. 项目结构

```text
insect_pest_complete/
├── classification/
│   ├── train.py
│   ├── compare.py
│   ├── predict.py
│   ├── batch_predict.py
│   └── video_predict.py
├── detection/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── attention_map.py
│   └── gradcam.py
├── configs/
│   └── yolo_data.example.yaml
├── requirements.txt
├── README_ZH.md
└── README_EN.md
```

## 3. 分类数据准备

分类模块支持两种 IP102 数据结构。

### 官方划分文件格式

```text
IP102/
├── images/
├── train.txt
├── val.txt
└── test.txt
```

每个划分文件中一行对应一张图片：

```text
00001.jpg 0
00002.jpg 15
```

### ImageFolder 格式

```text
IP102/
├── train/类别名/*.jpg
├── val/类别名/*.jpg
└── test/类别名/*.jpg
```

两个分类模型必须使用完全相同的训练集、验证集和测试集划分，否则实验结果不能公平比较。

## 4. 训练 ResNet18 Baseline

请从项目根目录运行：

```bash
python classification/train.py \
  --data-dir /path/to/IP102 \
  --model resnet18 \
  --epochs 30 \
  --batch-size 64 \
  --seed 42 \
  --output-dir outputs/classification
```

显存不足时，可将 `--batch-size` 改为 32、16 或 8。

## 5. 训练 ConvNeXt-Tiny

```bash
python classification/train.py \
  --data-dir /path/to/IP102 \
  --model convnext_tiny \
  --epochs 30 \
  --batch-size 64 \
  --seed 42 \
  --output-dir outputs/classification
```

ConvNeXt-Tiny 通常比 ResNet18 占用更多显存，可以为它设置较小的批大小，但数据划分、输入尺寸、增强、训练轮数和评价方法需要保持一致。

## 6. 生成分类对比结果

两个模型训练完成后运行：

```bash
python classification/compare.py --output-dir outputs/classification
```

程序会生成：

```text
outputs/classification/
├── resnet18/
│   ├── best.pt
│   ├── metrics.json
│   ├── history.json
│   ├── curves.png
│   └── confusion_matrix.png
├── convnext_tiny/
│   └── ...
├── comparison.csv
└── comparison.png
```

对比指标包括测试集 Accuracy、Macro-F1、参数量和单张图片推理延迟。

## 7. 分类推理

### 单张图片分类

```bash
python classification/predict.py \
  --checkpoint outputs/classification/convnext_tiny/best.pt \
  --image insect.jpg \
  --top-k 5
```

### 批量图片分类

```bash
python classification/batch_predict.py \
  --checkpoint outputs/classification/convnext_tiny/best.pt \
  --input-dir images \
  --output-csv outputs/classification/batch.csv \
  --batch-size 64 \
  --top-k 5 \
  --confidence 0.5
```

### 视频整帧分类

```bash
python classification/video_predict.py \
  --checkpoint outputs/classification/convnext_tiny/best.pt \
  --input-video input.mp4 \
  --output-video outputs/classification/video.mp4 \
  --confidence 0.5
```

分类视频功能只判断每一帧中最主要的昆虫类别，不会绘制目标框。如需在同一帧中定位多只昆虫，请使用 YOLO11 检测模块。

## 8. YOLO11 检测数据准备

YOLO11 必须使用带边界框的标注数据，只有图片类别的 IP102 数据不能直接训练检测模型。

推荐的数据结构：

```text
yolo_insect_dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

每张图片需要一个同名的 `.txt` 标签文件。每行标注格式为：

```text
class_id x_center y_center width height
```

坐标必须归一化到 0–1。例如：

```text
0 0.512 0.438 0.231 0.305
```

复制配置示例：

```bash
cp configs/yolo_data.example.yaml configs/yolo_data.yaml
```

然后修改数据根目录和真实类别名称。

## 9. 训练 YOLO11

```bash
python detection/train.py \
  --data configs/yolo_data.yaml \
  --model yolo11n.pt \
  --epochs 70 \
  --batch 16 \
  --imgsz 640
```

默认使用体积较小、速度较快的 YOLO11n。需要更高精度并且显存充足时，可以改为 `yolo11s.pt` 或 `yolo11m.pt`。

最佳权重通常保存在：

```text
runs/detect/yolo11_pest/weights/best.pt
```

## 10. 评估 YOLO11

验证集：

```bash
python detection/evaluate.py \
  --model runs/detect/yolo11_pest/weights/best.pt \
  --data configs/yolo_data.yaml \
  --split val
```

测试集：

```bash
python detection/evaluate.py \
  --model runs/detect/yolo11_pest/weights/best.pt \
  --data configs/yolo_data.yaml \
  --split test
```

主要指标包括：

- mAP50–95
- mAP50
- mAP75
- Precision
- Recall

程序还会生成混淆矩阵、PR 曲线等 Ultralytics 评估图表，以及 `metrics_summary.json`。

## 11. 图片、批量和视频检测

### 单张图片

```bash
python detection/predict.py \
  --model runs/detect/yolo11_pest/weights/best.pt \
  --source insect.jpg \
  --output-dir outputs/detection \
  --name single_image
```

### 批量目录

```bash
python detection/predict.py \
  --model runs/detect/yolo11_pest/weights/best.pt \
  --source images \
  --output-dir outputs/detection \
  --name batch_images
```

### 视频

```bash
python detection/predict.py \
  --model runs/detect/yolo11_pest/weights/best.pt \
  --source input.mp4 \
  --output-dir outputs/detection \
  --name video
```

### 摄像头

```bash
python detection/predict.py \
  --model runs/detect/yolo11_pest/weights/best.pt \
  --source 0 \
  --output-dir outputs/detection \
  --name webcam \
  --show
```

检测目录中会保存：

- 带目标框的图片或视频
- YOLO `.txt` 检测结果
- 包含类别、置信度和边界框坐标的 `detections.csv`

## 12. Attention Map

```bash
python detection/attention_map.py \
  --model runs/detect/yolo11_pest/weights/best.pt \
  --source images \
  --output-dir outputs/attention
```

程序会生成特征注意力叠加图和 `roi_scores.csv`。该注意力图来自 YOLO 中间特征的通道平均，适合用于辅助观察，不应被描述为严格的因果解释。

## 13. Grad-CAM

Grad-CAM 使用 YOLO 检测框裁剪昆虫区域，再利用本项目训练好的 ResNet18 或 ConvNeXt-Tiny 解释区域分类结果：

```bash
python detection/gradcam.py \
  --detector runs/detect/yolo11_pest/weights/best.pt \
  --classifier outputs/classification/convnext_tiny/best.pt \
  --image insect.jpg \
  --output outputs/gradcam.jpg
```

需要明确：这个 Grad-CAM 解释的是检测框内分类模型的判断，不是 YOLO 检测器本身的决策。

## 14. 公平实验原则

- ResNet18 与 ConvNeXt-Tiny 使用相同的数据划分。
- 两个分类模型使用相同输入尺寸、增强方式、训练轮数、早停策略和测试集。
- 计算资源允许时，每个模型使用随机种子 42、43、44 分别训练，报告平均值 ± 标准差。
- 保存每次运行的完整命令、GPU 型号、Python/PyTorch/torchvision 版本和训练时间。
- 分类结果单独报告 Accuracy 和 Macro-F1。
- 检测结果单独报告 mAP、Precision 和 Recall。
- 不要直接比较分类 Accuracy 和检测 mAP，因为它们回答的是不同问题。

## 15. 团队协作建议

推荐分工：

- 队员 A：数据下载、清理、划分和版本记录。
- 队员 B：ResNet18 Baseline 训练。
- 队员 C：ConvNeXt-Tiny 训练与分类对比。
- 队员 D：YOLO11 检测、视频演示和论文结果整理。

需要共享的实验文件：

- `metrics.json` 或 `metrics_summary.json`
- `history.json`
- `comparison.csv`
- 训练曲线、混淆矩阵和 PR 曲线
- 运行命令和环境版本
- `best.pt` 权重文件（建议通过网盘共享，不要直接提交到 Git）

## 16. 输出目录示例

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

