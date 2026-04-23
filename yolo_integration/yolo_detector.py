from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None


@dataclass
class Detection:
    xyxy: Tuple[int, int, int, int]  # (x0, y0, x1, y1)
    conf: float


class YoloShipDetector:
    """Small wrapper so we don't touch original codebase."""

    def __init__(self, weights_path: str, device: str | None = None):
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. Run: pip install -r yolo_integration/requirements_yolo.txt"
            )
        self.model = YOLO(weights_path)
        self.device = device

    @staticmethod
    def _tif2rgb_uint8(stacked_hw2: np.ndarray) -> np.ndarray:
        """Convert (H,W,2) float32 tif into (H,W,3) uint8 for YOLO.

        We build a pseudo-RGB:
        - R = VH
        - G = VV
        - B = VH - VV
        Then min-max normalize to [0,255].
        """
        if stacked_hw2.ndim != 3 or stacked_hw2.shape[-1] != 2:
            raise ValueError(f"Expected (H,W,2), got {stacked_hw2.shape}")

        vh = stacked_hw2[:, :, 0]
        vv = stacked_hw2[:, :, 1]
        b = vh - vv
        rgb = np.stack([vh, vv, b], axis=-1).astype(np.float32)

        mn = float(rgb.min())
        mx = float(rgb.max())
        rgb = (rgb - mn) / (mx - mn + 1e-6)
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        return rgb

    def detect(self, stacked_hw2: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> List[Detection]:
        rgb = self._tif2rgb_uint8(stacked_hw2)
        results = self.model.predict(rgb, conf=conf, iou=iou, verbose=False, device=self.device)

        dets: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
                dets.append(Detection((x0, y0, x1, y1), float(b.conf[0].cpu().item())))
        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets
