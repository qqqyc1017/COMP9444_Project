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

## 1. 下载官方 IP102 数据集

IP102 官方 GitHub：

```text
https://github.com/xpwu95/IP102
```

官方 Google Drive(推荐使用)：

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

