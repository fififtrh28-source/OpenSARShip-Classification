"""Export a YOLO-format dataset using CA-CFAR bbox as pseudo-labels.

Why this exists:
- Your current project is **classification** on 64x64 patches.
- To add YOLO safely, we create a *separate* dataset for detection.
- We reuse the existing CA-CFAR idea to generate bbox labels without hand-annotation.

Output structure:
  out_dir/
    images/train|val|test/*.png
    labels/train|val|test/*.txt
    data.yaml

Label format: one class only => 0 (ship)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import tifffile as tiff
from skimage import measure
from scipy.ndimage import label as scipy_label
import cv2


def ca_cfar_detection(image: np.ndarray, pfa: float = 1e-3) -> np.ndarray:
    """CA-CFAR like your dataload.py. Returns boolean mask."""
    N = image.size
    alpha = N * (pfa ** (-1 / N) - 1)
    background_mean = float(image.mean())
    threshold = alpha * background_mean
    return image > threshold


def cfar_bbox(stacked_hw2: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x0,y0,x1,y1) bbox of largest region detected on VH channel."""
    vh = stacked_hw2[:, :, 0]
    mask = ca_cfar_detection(vh)
    labeled, n = scipy_label(mask)
    if n <= 0:
        return None

    props = measure.regionprops(labeled, intensity_image=vh)
    ship = max(props, key=lambda x: x.area)
    y0, x0, y1, x1 = ship.bbox

    # Guard against tiny boxes
    if (x1 - x0) < 2 or (y1 - y0) < 2:
        return None

    return (int(x0), int(y0), int(x1), int(y1))


def tif2rgb_uint8(stacked_hw2: np.ndarray) -> np.ndarray:
    """Same pseudo-RGB as in yolo_detector.py, but exported as PNG for YOLO training."""
    vh = stacked_hw2[:, :, 0]
    vv = stacked_hw2[:, :, 1]
    b = vh - vv
    rgb = np.stack([vh, vv, b], axis=-1).astype(np.float32)
    mn, mx = float(rgb.min()), float(rgb.max())
    rgb = (rgb - mn) / (mx - mn + 1e-6)
    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


def write_yolo_label(label_path: Path, bbox_xyxy: tuple[int, int, int, int], w: int, h: int):
    x0, y0, x1, y1 = bbox_xyxy
    xc = ((x0 + x1) / 2.0) / w
    yc = ((y0 + y1) / 2.0) / h
    bw = (x1 - x0) / w
    bh = (y1 - y0) / h

    # class 0 = ship
    label_path.write_text(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True, help="Folder containing .tif (H,W,2)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1.0")

    random.seed(args.seed)

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    tif_paths = sorted(images_dir.glob("*.tif"))
    if not tif_paths:
        raise FileNotFoundError(f"No .tif found in {images_dir}")

    random.shuffle(tif_paths)

    n = len(tif_paths)
    n_train = int(n * args.train)
    n_val = int(n * args.val)

    splits = {
        "train": tif_paths[:n_train],
        "val": tif_paths[n_train : n_train + n_val],
        "test": tif_paths[n_train + n_val :],
    }

    # Create folders
    for sp in ["train", "val", "test"]:
        (out_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0

    for sp, paths in splits.items():
        for p in paths:
            arr = tiff.imread(str(p)).astype(np.float32)
            if arr.ndim != 3 or arr.shape[-1] != 2:
                skipped += 1
                continue

            bbox = cfar_bbox(arr)
            if bbox is None:
                skipped += 1
                continue

            h, w = arr.shape[:2]
            rgb = tif2rgb_uint8(arr)

            # Save as PNG (Ultralytics YOLO handles PNG well)
            img_out = out_dir / "images" / sp / (p.stem + ".png")
            # cv2 expects BGR
            cv2.imwrite(str(img_out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            label_out = out_dir / "labels" / sp / (p.stem + ".txt")
            write_yolo_label(label_out, bbox, w=w, h=h)

            kept += 1

    # data.yaml
    yaml = f"""path: {out_dir.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: ship
"""
    (out_dir / "data.yaml").write_text(yaml)

    print(f"Done. kept={kept}, skipped={skipped}, total={n}")
    print(f"YOLO data.yaml written to: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
