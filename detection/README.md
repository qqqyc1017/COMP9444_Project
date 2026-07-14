# YOLO11 detection module

This module is a cleaned and parameterised integration of the previous YOLO pest-detection work. The original ideas—training, evaluation, prediction, feature attention and Grad-CAM—are retained, while hard-coded personal paths and GUI-only behaviour have been removed.

The detector requires YOLO bounding-box labels. Classification-only IP102 labels are not sufficient.

```text
dataset/
├── train/{images,labels}
├── valid/{images,labels}
├── test/{images,labels}
└── data.yaml
```

Each label line is `class_id x_center y_center width height`, normalised to 0–1.

Common commands are documented in the project-level README.

