# IP102 昆虫识别完整项目

本项目用于完成昆虫害虫识别任务，包含两条功能路线：

1. **图像分类**：训练 ResNet18 Baseline 和 ConvNeXt-Tiny，并进行公平对比。
2. **视频目标检测**：使用已有的 YOLO11 `best.pt`，在视频中识别并定位多只昆虫。

分类和检测的区别：

- 分类模型判断整张图片或整帧视频的主要昆虫类别。
- YOLO11 检测模型可以识别多只昆虫，并绘制边界框。

## 1. 项目结构

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
├── README.md
├── README_ZH.md
└── README_EN.md
```

## 2. 环境安装

推荐使用 Python 3.10 或 Python 3.11。

创建虚拟环境：

```bash
python -m venv .venv
```

macOS/Linux：

```bash
source .venv/bin/activate
```

Windows PowerShell：

```powershell
.venv\Scripts\activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

检查 PyTorch 和 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

NVIDIA GPU 正常配置时，最后一项应输出 `True`。没有 GPU 也可以运行，但训练速度会明显变慢。

## 3. 下载 IP102 分类数据集

IP102 官方 GitHub：

```text
https://github.com/xpwu95/IP102
```

官方 Google Drive：

```text
https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo?usp=sharing
```

阿里云盘备用入口：

```text
https://www.aliyundrive.com/s/c5G9scSGyak
```

进入 Google Drive 的 `IP102_v1.1` 根目录后，选择：

```text
Classification/
```

通常需要下载：

```text
Classification/
├── ip102_v1.1.tar
└── classes.txt
```

如果只看到 `resnet50_0.497.pkl`，说明进入了 `IP102_pretrained_models` 子文件夹。该文件不是数据集，请返回 `IP102_v1.1` 根目录并进入 `Classification`。

## 4. 解压和定位分类数据

macOS/Linux：

```bash
mkdir -p datasets/IP102
tar -xf /path/to/ip102_v1.1.tar -C datasets/IP102
```

Windows PowerShell：

```powershell
mkdir datasets\IP102
tar -xf C:\path\to\ip102_v1.1.tar -C datasets\IP102
```

解压后搜索：

```text
train.txt
val.txt
test.txt
```

`--data-dir` 应指向实际包含这些 split 文件的目录。例如：

```text
datasets/IP102/ip102_v1.1/
├── images/
├── train.txt
├── val.txt
├── test.txt
└── classes.txt
```

对应参数：

```bash
--data-dir datasets/IP102/ip102_v1.1
```

项目也支持 ImageFolder 格式：

```text
IP102/
├── train/类别名/*.jpg
├── val/类别名/*.jpg
└── test/类别名/*.jpg
```

## 5. Classification 文件夹代码说明

### `train.py`——分类训练入口

负责训练 ResNet18 或 ConvNeXt-Tiny，包含：

- 数据集加载
- ImageNet 迁移学习
- AdamW 优化器
- 余弦学习率调度
- 标签平滑
- Early Stopping
- 最佳模型保存
- 测试集评估
- 训练曲线和混淆矩阵输出

### `models.py`——模型定义

负责创建：

- `resnet18`：Baseline，模型较小、速度较快。
- `convnext_tiny`：先进模型，通常具有更强的特征提取能力。

程序会加载 ImageNet 预训练权重，并将最终分类层替换成 IP102 类别输出层。

### `data.py`——数据读取和增强

负责读取官方 split 或 ImageFolder 数据。训练增强包括：

- Random Resized Crop
- 随机水平翻转
- RandAugment
- ImageNet 标准化
- Random Erasing

验证和测试只使用固定缩放、中心裁剪和标准化。

### `engine.py`——公共训练引擎

负责：

- 设置随机种子
- 自动选择 CUDA、Apple MPS 或 CPU
- 执行训练、验证和测试 epoch
- 计算 Loss、Accuracy 和 Macro-F1
- 测量推理延迟
- 绘制训练曲线和混淆矩阵
- 保存 JSON 实验结果

两个模型使用相同引擎，可以保证实验流程一致。

### `compare.py`——分类模型对比

读取：

```text
outputs/classification/resnet18/metrics.json
outputs/classification/convnext_tiny/metrics.json
```

生成：

```text
outputs/classification/comparison.csv
outputs/classification/comparison.png
```

对比 Accuracy、Macro-F1、参数量和单张图片推理延迟。

### `inference_utils.py`——推理公共功能

负责加载 checkpoint、类别名称、图片预处理和运行设备，供批量分类和视频分类调用。

### `predict.py`——单张图片分类

输入一张图片，输出 Top-K 昆虫类别和置信度。

### `batch_predict.py`——批量图片分类

递归扫描图片目录，批量推理并将类别和置信度导出到 CSV。

### `video_predict.py`——视频整帧分类

逐帧分类并在视频左上角显示主要类别和置信度。该脚本不绘制目标框。

### 模块调用关系

```text
data.py ────────────┐
models.py ──────────┼→ train.py → best.pt + metrics.json
engine.py ──────────┘                   │
                                       ├→ predict.py
inference_utils.py ←────────────────────┼→ batch_predict.py
                                       └→ video_predict.py

resnet18/metrics.json ─────┐
                           ├→ compare.py → comparison.csv + comparison.png
convnext_tiny/metrics.json ┘
```

## 6. 训练前快速测试

正式训练前先运行 1 个 epoch：

```bash
python classification/train.py \
  --data-dir /path/to/IP102 \
  --model resnet18 \
  --epochs 1 \
  --batch-size 8 \
  --workers 0 \
  --output-dir outputs/smoke_test
```

快速测试可以完成后再开始正式训练。

## 7. 训练 ResNet18 Baseline

```bash
python classification/train.py \
  --data-dir /path/to/IP102 \
  --model resnet18 \
  --epochs 30 \
  --batch-size 64 \
  --seed 42 \
  --output-dir outputs/classification
```

显存不足时可将 batch size 改为 32、16 或 8。

## 8. 训练 ConvNeXt-Tiny

```bash
python classification/train.py \
  --data-dir /path/to/IP102 \
  --model convnext_tiny \
  --epochs 30 \
  --batch-size 32 \
  --seed 42 \
  --output-dir outputs/classification
```

两个模型必须使用相同的数据划分、输入尺寸、增强方式、训练轮数、随机种子和测试集。

## 9. 生成分类对比结果

```bash
python classification/compare.py --output-dir outputs/classification
```

典型输出：

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

## 10. 单张图片分类

```bash
python classification/predict.py \
  --checkpoint outputs/classification/convnext_tiny/best.pt \
  --image insect.jpg \
  --top-k 5
```

## 11. 批量图片分类

```bash
python classification/batch_predict.py \
  --checkpoint outputs/classification/convnext_tiny/best.pt \
  --input-dir images \
  --output-csv outputs/classification/batch.csv \
  --batch-size 64 \
  --top-k 5 \
  --confidence 0.5
```

## 12. 视频整帧分类

```bash
python classification/video_predict.py \
  --checkpoint outputs/classification/convnext_tiny/best.pt \
  --input-video input.mp4 \
  --output-video outputs/classification/video.mp4 \
  --confidence 0.5 \
  --frame-step 1
```

如果处理速度较慢，可使用 `--frame-step 3`，每三帧运行一次模型。

## 13. 使用已有 YOLO11 best.pt 检测视频

需要准备：

- 一个已经训练完成的 YOLO11 检测模型 `best.pt`
- 一个输入视频，例如 `input.mp4`

不需要重新训练 YOLO，也不需要下载 IP102 Detection 数据。

### 检查模型类型和类别

```bash
python -c "from ultralytics import YOLO; m=YOLO('/path/to/best.pt'); print('task:', m.task); print('classes:', m.names)"
```

模型任务必须为：

```text
task: detect
```

如果是 `classify`，则不能绘制检测框。

### 运行视频检测

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

设备设置：

```text
--device 0      NVIDIA GPU
--device mps    Apple Silicon
--device cpu    CPU
```

不设置 `--device` 时由 Ultralytics 自动选择设备。

### YOLO 视频输出

```text
outputs/
├── yolo_video.mp4
└── yolo_detections.csv
```

视频中会显示：

- 昆虫边界框
- 类别名称
- 检测置信度

CSV 会记录：

- 帧号
- 视频时间
- 类别编号和名称
- 置信度
- `x1, y1, x2, y2` 边界框坐标

OpenCV 生成的视频通常不保留原始音轨。如果需要音频，可以使用 FFmpeg 将原视频音频合并回检测视频。

## 14. 两种视频程序的区别

| 程序 | 使用模型 | 功能 |
|---|---|---|
| `classification/video_predict.py` | ResNet18 或 ConvNeXt-Tiny | 每帧一个主要类别，不绘制框 |
| `yolo_video/detect_video.py` | 已训练的 YOLO11 `best.pt` | 检测多只昆虫并绘制边界框 |

## 15. 公平实验与团队协作

- 两个分类模型使用完全相同的 train/val/test 划分。
- 保存每次实验的运行命令、随机种子、GPU 型号和训练时间。
- 计算资源允许时，使用随机种子 42、43、44 分别训练三次。
- 报告 Accuracy 和 Macro-F1 的平均值 ± 标准差。
- 共享 `metrics.json`、`history.json`、曲线、混淆矩阵和对比 CSV。
- `.pt` 权重建议通过 Google Drive 或 OneDrive 分享，不提交到 Git。

