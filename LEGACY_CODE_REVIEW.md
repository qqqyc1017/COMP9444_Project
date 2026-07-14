# Review of the previous YOLO scripts

- `train.py` described YOLOv8 and loaded `yolov8m.pt`; the integrated version explicitly defaults to `yolo11n.pt` and exposes all important settings as arguments.
- Several scripts contained absolute Windows/macOS user paths. All paths are now command-line arguments.
- The previous prediction script supported only one image and always opened a GUI window. The new version accepts images, directories, videos, streams and webcams, saves annotations/labels, and exports a CSV.
- The previous evaluation included a custom confidence-based ROC-AUC calculation. ROC-AUC is not a standard primary object-detection metric and can be misleading. The integrated evaluator reports Ultralytics mAP50–95, mAP50, mAP75, precision and recall.
- The attention script could divide by zero for constant feature maps, leak the CSV handle on early return, and assumed a tensor output from a fixed layer. These cases are handled explicitly.
- The previous Grad-CAM script applied an unrelated ImageNet ResNet18 to YOLO crops. The new script requires this project's trained pest classifier and clearly states that the map explains the crop classifier, not YOLO itself.

