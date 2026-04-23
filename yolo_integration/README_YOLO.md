# YOLO integration (non-breaking)

This folder adds YOLO ship **detection** as an optional pre-step before the existing **classification** pipeline.
Nothing in the original codebase is modified.

## Install (recommended: separate venv)
```bash
python -m venv .venv_yolo
source .venv_yolo/bin/activate  # Windows: .venv_yolo\\Scripts\\activate

pip install -r ../requirements.txt
pip install -r requirements_yolo.txt
```

## 1) Export pseudo-labels (bbox) for YOLO using existing CA-CFAR
This converts your `resized_new/*.tif` patches into a YOLO dataset using CA-CFAR bbox as labels.

```bash
python export_yolo_dataset_from_cfar.py \
  --images_dir ../resized_new \
  --out_dir ./yolo_dataset \
  --train 0.8 --val 0.1 --test 0.1
```

This creates:
- `yolo_dataset/images/{train,val,test}`
- `yolo_dataset/labels/{train,val,test}`
- `yolo_dataset/data.yaml`

## 2) Train YOLO (Ultralytics)
Example (YOLOv8n):
```bash
yolo detect train data=yolo_dataset/data.yaml model=yolov8n.pt imgsz=256 epochs=50
```

## 3) Detect then classify (end-to-end)
Run YOLO on a tif patch (H,W,2), crop ship bbox, resize to 64x64x2, standardize, then classify.

```bash
python detect_and_classify.py \
  --yolo_weights /path/to/best.pt \
  --clf_weights /path/to/experiments/.../model.pth \
  --image /path/to/sample.tif
```

Notes:
- YOLO is trained as **1 class**: `ship`.
- Classification uses your existing model (`ResNet50WithFeatures`, etc.).
