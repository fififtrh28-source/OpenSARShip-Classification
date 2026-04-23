"""End-to-end: YOLO detect -> crop -> classification.

This script does NOT modify any existing training code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import tifffile as tiff
import torch
from skimage.transform import resize
from skimage import measure
from scipy.ndimage import label as scipy_label

from yolo_detector import YoloShipDetector

# Import your existing classifier architectures
from model import ResNet50WithFeatures, BaselineModel, AlexNetWithFeatures, VGG19WithFeatures


CLASS_NAMES = {
    0: "Bulk Carrier",
    1: "Container Ship",
    2: "Fishing",
    3: "Tanker",
}


def ca_cfar_detection(image: np.ndarray, pfa: float = 1e-3) -> np.ndarray:
    """CA-CFAR like your dataload.py. Returns boolean mask."""
    N = image.size
    alpha = N * (pfa ** (-1 / N) - 1)
    background_mean = float(image.mean())
    threshold = alpha * background_mean
    return image > threshold


def extract_scale_variant_features(stacked_hw2: np.ndarray, incidence_angle: float) -> np.ndarray:
    """Copy of OpenSARShipDataset.extract_scale_variant_features.

    We keep it local so this script doesn't require the full dataset folder.
    """
    features = np.zeros(14, dtype=np.float32)
    features[0] = float(incidence_angle)

    vh_img = stacked_hw2[:, :, 0]
    binary_mask = ca_cfar_detection(vh_img)
    labeled_img, num_features_detected = scipy_label(binary_mask)

    if num_features_detected > 0:
        props = measure.regionprops(labeled_img, intensity_image=stacked_hw2)
        ship = max(props, key=lambda x: x.area)

        y0, x0, y1, x1 = ship.bbox
        features[1] = max(y1 - y0, x1 - x0)
        features[2] = ship.area
        features[3] = vh_img[labeled_img == ship.label].sum()
        features[4] = ship.bbox_area

        ship_pixels = vh_img[labeled_img == ship.label].astype(np.float32)
        background_pixels = vh_img[labeled_img != ship.label].astype(np.float32)

        m_t = float(ship_pixels.mean()) if ship_pixels.size else 0.0
        m_max = float(ship_pixels.max()) if ship_pixels.size else 0.0
        m_b = float(background_pixels.mean()) if background_pixels.size else 0.0
        s_b = float(background_pixels.std()) if background_pixels.size else 0.0

        if background_pixels.size and s_b > 0:
            features[5] = (m_t - m_b) / s_b
            features[6] = (m_max - m_b) / s_b

        hu_moments = ship.moments_hu
        features[7:14] = hu_moments

    return features


def standardize(stacked_hw2: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (stacked_hw2.astype(np.float32) - mean) / (std + 1e-6)


def crop_and_resize(stacked_hw2: np.ndarray, xyxy: tuple[int, int, int, int], out_size: int = 64) -> np.ndarray:
    x0, y0, x1, y1 = xyxy
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(stacked_hw2.shape[1], x1)
    y1 = min(stacked_hw2.shape[0], y1)
    crop = stacked_hw2[y0:y1, x0:x1, :]
    if crop.size == 0:
        raise ValueError("Empty crop from bbox")

    resized = resize(crop, (out_size, out_size), order=3, mode="reflect", preserve_range=True).astype(np.float32)
    return resized


def load_classifier(model_type: str, ckpt_path: str, device: str):
    if model_type == "resnet50":
        model = ResNet50WithFeatures(num_features=14, num_classes=4, pretrained=False)
    elif model_type == "baseline":
        model = BaselineModel(num_classes=4, num_features=14)
    elif model_type == "alexnet":
        model = AlexNetWithFeatures(num_features=14, num_classes=4, pretrained=False)
    elif model_type == "vgg19":
        model = VGG19WithFeatures(num_features=14, num_classes=4, pretrained=False)
    else:
        raise ValueError("model_type must be one of: resnet50, baseline, alexnet, vgg19")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_weights", type=str, required=True)
    ap.add_argument("--clf_weights", type=str, required=True)
    ap.add_argument("--model_type", type=str, default="resnet50")
    ap.add_argument("--image", type=str, required=True, help=".tif with shape (H,W,2)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--std_params", type=str, default="../standardization_params.npz")
    ap.add_argument("--feature_scaler", type=str, default="../feature_scaler.save")
    ap.add_argument("--incidence", type=float, default=0.0, help="If unknown, keep 0")
    args = ap.parse_args()

    img = tiff.imread(args.image).astype(np.float32)
    if img.ndim != 3 or img.shape[-1] != 2:
        raise ValueError(f"Expected (H,W,2) tif, got {img.shape}")

    detector = YoloShipDetector(args.yolo_weights, device=args.device)
    dets = detector.detect(img, conf=args.conf, iou=args.iou)
    if not dets:
        print("No ship detected.")
        return

    best = dets[0]
    crop64 = crop_and_resize(img, best.xyxy, out_size=64)

    # Standardize using training-set stats
    std_params = np.load(args.std_params)
    mean = float(std_params["mean"])
    std = float(std_params["std"])
    crop64 = standardize(crop64, mean=mean, std=std)

    # Compute 14 features (same logic as dataload.py)
    feats = extract_scale_variant_features(crop64, incidence_angle=args.incidence)

    # Scale features to match training
    scaler = joblib.load(args.feature_scaler)
    feats_scaled = scaler.transform([feats])[0].astype(np.float32)

    # Torch tensors
    x_img = torch.from_numpy(crop64).float().permute(2, 0, 1).unsqueeze(0).to(args.device)  # (1,2,64,64)
    x_feat = torch.from_numpy(feats_scaled).float().unsqueeze(0).to(args.device)  # (1,14)

    clf = load_classifier(args.model_type, args.clf_weights, device=args.device)
    with torch.no_grad():
        logits = clf(x_img, x_feat)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())

    print(f"YOLO bbox={best.xyxy}, conf={best.conf:.3f}")
    print("Classification probabilities:")
    for i, p in enumerate(probs):
        print(f"  {i}: {CLASS_NAMES[i]} = {p*100:.2f}%")
    print(f"\nPREDICTED: {pred} ({CLASS_NAMES[pred]})")


if __name__ == "__main__":
    main()
