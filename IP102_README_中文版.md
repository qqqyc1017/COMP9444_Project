# IP102 昆虫识别项目——中文使用说明

本项目用于完成以下任务：

1. 使用 ResNet18 作为 Baseline 进行 IP102 害虫分类。
2. 使用 ConvNeXt-Tiny 作为先进模型进行分类，并与 ResNet18 对比。
3. 对单张图片、批量图片和视频进行分类推理。
4. 使用带边界框标注的 IP102 Detection 数据训练 YOLO11，实现图片和视频目标检测。

分类与检测是两个不同任务：

- 分类评价指标：Accuracy、Macro-F1、参数量、推理延迟。
- 检测评价指标：mAP50–95、mAP50、Precision、Recall。
- 不要直接比较分类 Accuracy 和检测 mAP。

## Classification 文件夹代码说明

`classification` 文件夹负责“整张图片属于哪一种昆虫”的分类任务，不负责绘制昆虫位置框。它包含数据读取、模型定义、训练评估、模型对比以及图片和视频推理的完整流程。

### `train.py`——分类模型训练入口

负责训练 ResNet18 或 ConvNeXt-Tiny，主要功能包括：

- 读取训练参数。
- 加载 IP102 训练集、验证集和测试集。
- 创建指定分类模型。
- 使用 AdamW 优化器训练。
- 使用余弦学习率调度。
- 使用标签平滑减少过拟合。
- 根据验证集 Accuracy 保存最佳模型。
- 使用 Early Stopping 提前停止无效训练。
- 在测试集上计算 Loss、Accuracy 和 Macro-F1。
- 输出训练曲线、混淆矩阵和 JSON 指标。

训练 ResNet18：

```bash
python classification/train.py \
  --data-dir /path/to/IP102 \
  --model resnet18 \
  --epochs 30
```

训练 ConvNeXt-Tiny：

```bash
python classification/train.py \
  --data-dir /path/to/IP102 \
  --model convnext_tiny \
  --epochs 30
```

### `models.py`——模型定义

负责创建两个分类模型：

- `resnet18`：作为 Baseline，模型较小、训练和推理速度较快。
- `convnext_tiny`：作为先进模型，通常具有更强的特征提取能力。

程序默认加载 ImageNet 预训练权重，并将模型原来的 1000 类输出层替换成 IP102 的类别输出层。它还负责统计模型的可训练参数量。

整体过程为：

```text
输入图片 → ResNet18 或 ConvNeXt-Tiny → IP102 类别概率
```

### `data.py`——数据读取和增强

负责将 IP102 图片转换为 PyTorch 可以训练的数据，支持以下两种结构：

官方 split 文件格式：

```text
IP102/
├── images/
├── train.txt
├── val.txt
└── test.txt
```

ImageFolder 格式：

```text
IP102/
├── train/类别名/*.jpg
├── val/类别名/*.jpg
└── test/类别名/*.jpg
```

训练阶段的数据增强包括：

- Random Resized Crop
- 随机水平翻转
- RandAugment
- ImageNet 标准化
- Random Erasing

验证和测试阶段只使用固定缩放、中心裁剪和标准化，不使用随机增强。

### `engine.py`——训练和评估核心逻辑

这是两个分类模型共用的训练引擎，主要功能包括：

- 设置 Python、NumPy 和 PyTorch 随机种子。
- 自动选择 NVIDIA CUDA、Apple MPS 或 CPU。
- 执行一个训练、验证或测试 epoch。
- 计算 Loss、Accuracy 和 Macro-F1。
- 测量 batch size 为 1 时的推理延迟。
- 绘制训练 Loss 和 Accuracy 曲线。
- 绘制归一化混淆矩阵。
- 保存 JSON 格式的实验结果。

两个模型使用相同的 `engine.py`，可以减少训练流程差异，使对比实验更加公平。

### `compare.py`——模型对比

负责读取两个模型训练产生的：

```text
outputs/resnet18/metrics.json
outputs/convnext_tiny/metrics.json
```

然后生成：

```text
outputs/comparison.csv
outputs/comparison.png
```

对比项目包括：

- 测试集 Accuracy
- 测试集 Macro-F1
- 可训练参数量
- 单张图片推理延迟

必须先完成两个模型训练，再运行：

```bash
python classification/compare.py --output-dir outputs
```

### `inference_utils.py`——推理公共工具

为批量图片、视频和 Grad-CAM 提供公共功能，主要负责：

- 从 `best.pt` 加载模型权重。
- 恢复模型名称和类别列表。
- 加载与训练一致的图片预处理。
- 自动选择 GPU、MPS 或 CPU。
- 对一批图片执行 Top-K 分类。

这样不同推理脚本不需要重复编写模型加载代码。

### `predict.py`——单张图片分类

输入一张图片，输出最可能的 Top-K 昆虫类别及置信度：

```bash
python classification/predict.py \
  --checkpoint outputs/convnext_tiny/best.pt \
  --image insect.jpg \
  --top-k 5
```

输出示例：

```text
rice leaf roller       82.31%
rice leaf caterpillar  10.42%
corn borer              2.15%
```

### `batch_predict.py`——批量图片分类

递归扫描目录中的图片，并使用 batch 一次处理多张图片。支持 JPG、PNG、BMP、WebP 和 TIFF，可设置 Top-K 和置信度阈值，并将结果导出为 CSV。

```bash
python classification/batch_predict.py \
  --checkpoint outputs/convnext_tiny/best.pt \
  --input-dir test_images \
  --output-csv outputs/predictions.csv \
  --batch-size 64 \
  --top-k 5 \
  --confidence 0.5
```

输出 CSV 示例：

```text
image,status,top1_class,top1_confidence
001.jpg,ok,rice leaf roller,0.9231
002.jpg,uncertain,corn borer,0.3842
```

损坏的图片会被记录为错误，但不会中断整个批量任务。

### `video_predict.py`——视频整帧分类

逐帧读取视频，将每一帧作为一张完整图片进行分类，并在画面左上角显示类别和置信度。

```bash
python classification/video_predict.py \
  --checkpoint outputs/convnext_tiny/best.pt \
  --input-video insects.mp4 \
  --output-video outputs/annotated.mp4 \
  --confidence 0.5 \
  --frame-step 1
```

设置 `--frame-step 3` 表示每三帧执行一次模型，中间帧复用最近结果，可以提高处理速度。

需要注意：这是整帧分类功能。如果一帧中同时出现多只或多种昆虫，它只能给出整帧的主要类别；需要分别绘制检测框时，应使用 YOLO11 检测模块。

### Classification 模块调用关系

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

## 1. 下载官方 IP102 数据集

IP102 官方 GitHub：

```text
https://github.com/xpwu95/IP102
```

官方 Google Drive（推荐）：

```text
https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo?usp=sharing
```

阿里云盘备用入口：

```text
https://www.aliyundrive.com/s/c5G9scSGyak
```

打开 Google Drive 后，应进入 `IP102_v1.1` 根目录，并看到：

```text
IP102_v1.1/
├── Classification/
└── Detection/
```

如果只看到：

```text
REDEME.txt
resnet50_0.497.pkl
```

说明进入了 `IP102_pretrained_models` 子文件夹。这里存放的是作者提供的旧 ResNet50 权重，不是数据集。请点击浏览器返回按钮，回到 `IP102_v1.1` 根目录。

## 2. 应该下载哪些文件

建议同时下载两个文件夹：

- `Classification`：用于训练 ResNet18 和 ConvNeXt-Tiny。
- `Detection`：用于训练 YOLO11。

进入文件夹后选择全部文件，点击下载按钮或右键选择“下载”。Google Drive 会先压缩文件夹，数据较大时需要等待。

常见下载内容：

```text
Classification/
├── ip102_v1.1.tar
└── classes.txt

Detection/
├── JPEGImages.tar
└── Annotations.tar
```

数据说明：

- 分类部分约有 75,222 张图片、102 个类别。
- 检测部分约有 18,981 张带边界框标注的图片。
- 数据集免费用于学术研究；其他用途请联系原作者。

## 3. 解压分类数据

### macOS/Linux

```bash
mkdir -p datasets/IP102_classification
tar -xf /path/to/ip102_v1.1.tar -C datasets/IP102_classification
```

### Windows PowerShell

```powershell
mkdir datasets\IP102_classification
tar -xf C:\path\to\ip102_v1.1.tar -C datasets\IP102_classification
```

解压后搜索以下划分文件：

```text
train.txt
val.txt
test.txt
```

训练程序的 `--data-dir` 应指向实际包含这些文件的目录，而不是一定指向最外层目录。

例如解压结果为：

```text
datasets/IP102_classification/ip102_v1.1/
├── images/
├── train.txt
├── val.txt
├── test.txt
└── classes.txt
```

则训练参数应写成：

```bash
--data-dir datasets/IP102_classification/ip102_v1.1
```

## 4. 检查分类数据

macOS/Linux：

```bash
find datasets/IP102_classification -name train.txt -o -name val.txt -o -name test.txt
```

查看图片数量：

```bash
find datasets/IP102_classification -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l
```

Windows 可以在文件资源管理器中搜索 `train.txt`，然后将该文件所在目录作为数据根目录。

## 5. 安装运行环境

推荐 Python 3.10 或 Python 3.11。

创建虚拟环境：

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

安装依赖：

```bash
pip install -r requirements.txt
```

检查 PyTorch 和 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

NVIDIA GPU 正常配置时，最后一项应输出 `True`。如果输出 `False`，程序仍可在 CPU 上运行，但训练速度会很慢。

## 6. 先进行一次快速测试

在正式训练前先运行 1 个 epoch，确认数据路径和环境正确：

```bash
python train.py \
  --data-dir datasets/IP102_classification/ip102_v1.1 \
  --model resnet18 \
  --epochs 1 \
  --batch-size 8 \
  --workers 0 \
  --output-dir outputs/smoke_test
```

如果实际数据根目录不同，请替换 `--data-dir`。

快速测试能完成后，再开始正式训练。

## 7. 训练 ResNet18 Baseline

```bash
python train.py \
  --data-dir datasets/IP102_classification/ip102_v1.1 \
  --model resnet18 \
  --epochs 30 \
  --batch-size 64 \
  --seed 42 \
  --output-dir outputs
```

显存不足时依次尝试：

```text
--batch-size 32
--batch-size 16
--batch-size 8
```

## 8. 训练 ConvNeXt-Tiny

```bash
python train.py \
  --data-dir datasets/IP102_classification/ip102_v1.1 \
  --model convnext_tiny \
  --epochs 30 \
  --batch-size 32 \
  --seed 42 \
  --output-dir outputs
```

ConvNeXt-Tiny 通常比 ResNet18 占用更多显存，因此可以使用较小的批大小。两个模型必须保持以下条件一致：

- 相同训练集、验证集和测试集。
- 相同输入尺寸。
- 相同数据增强。
- 相同训练轮数和早停策略。
- 相同随机种子。
- 相同测试集和评价指标。

## 9. 生成模型对比结果

两个模型训练完成后运行：

```bash
python compare.py --output-dir outputs
```

程序会生成：

```text
outputs/
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

论文中应使用 `comparison.csv` 的真实实验结果，不要提前填写或虚构准确率。

## 10. 单张图片分类

```bash
python predict.py \
  --checkpoint outputs/convnext_tiny/best.pt \
  --image insect.jpg \
  --top-k 5
```

## 11. 批量图片分类

```bash
python batch_predict.py \
  --checkpoint outputs/convnext_tiny/best.pt \
  --input-dir images \
  --output-csv outputs/batch_predictions.csv \
  --batch-size 64 \
  --top-k 5 \
  --confidence 0.5
```

程序会递归读取图片，并将预测类别和置信度保存到 CSV。

## 12. 视频整帧分类

```bash
python video_predict.py \
  --checkpoint outputs/convnext_tiny/best.pt \
  --input-video input.mp4 \
  --output-video outputs/annotated.mp4 \
  --confidence 0.5 \
  --frame-step 1
```

该功能对整帧进行分类，不会绘制昆虫位置框。如果同一帧中有多只昆虫，需要使用 YOLO11 检测模型。

## 13. 解压检测数据

### macOS/Linux

```bash
mkdir -p datasets/IP102_detection_raw
tar -xf /path/to/JPEGImages.tar -C datasets/IP102_detection_raw
tar -xf /path/to/Annotations.tar -C datasets/IP102_detection_raw
```

### Windows PowerShell

```powershell
mkdir datasets\IP102_detection_raw
tar -xf C:\path\to\JPEGImages.tar -C datasets\IP102_detection_raw
tar -xf C:\path\to\Annotations.tar -C datasets\IP102_detection_raw
```

解压后通常得到：

```text
datasets/IP102_detection_raw/
├── JPEGImages/
└── Annotations/
```

## 14. 检测标注格式说明

官方 `Annotations` 使用 Pascal VOC XML 格式，而 YOLO11 需要以下 TXT 格式：

```text
class_id x_center y_center width height
```

因此官方检测数据不能直接交给 YOLO11，必须先完成：

1. Pascal VOC XML 转 YOLO TXT。
2. 建立统一类别名称到类别编号的映射。
3. 划分训练集、验证集和测试集。
4. 生成 YOLO `data.yaml`。

转换后的推荐目录：

```text
IP102_detection_yolo/
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

如果不希望自行转换，可以使用已经整理为 YOLO 格式的社区镜像：

```text
https://www.kaggle.com/datasets/leonidkulyk/ip102-yolov5
```

虽然页面名称为 YOLOv5，但 YOLO11 使用相同的标签格式。下载后仍需要检查 `data.yaml` 中的路径和类别名称。

## 15. 团队实验要求

建议所有队员统一：

- 数据集版本。
- `train/val/test` 划分文件。
- 类别编号和类别名称。
- Python、PyTorch 和 torchvision 版本。
- 随机种子和训练参数。

每次训练后至少共享：

- `metrics.json`
- `history.json`
- `curves.png`
- `confusion_matrix.png`
- 实际运行命令
- GPU 型号和训练时间
- `best.pt`（建议通过网盘共享）

计算资源允许时，每个分类模型分别使用随机种子 42、43、44 训练三次，并报告平均值 ± 标准差。
