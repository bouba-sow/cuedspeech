"""Handshape images: anchor points, marker removal and compositing onto frames.

Each handshape image ``handshape_<n>.png`` (RGBA, transparent background,
fingers pointing up) carries a coloured marker (red or black) on the tip of
the reference finger. Two anchors are needed per image:

- ``target``: the reference fingertip, placed on the vowel position;
- ``wrist``: the middle of the top edge of the cuff, used for the hand size.

Anchors are read from ``handshapes.json`` in the same folder when present,
otherwise detected from the image. Run ``python -m cued_speech.hand_assets
<folder>`` to write that file and a preview image to check the anchors.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SHAPES = range(1, 9)
METADATA_FILE = "handshapes.json"

# Wrist -> middle-finger knuckle distance over wrist -> reference fingertip
# distance, for an average hand (Auto-CS default hand: 71 / 172).
KNUCKLE_TO_TIP_RATIO = 71.0 / 172.0


def _largest_component(mask: np.ndarray) -> Optional[np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == k


def _marker_mask(image: np.ndarray) -> np.ndarray:
    """Pixels of the red or black fingertip marker."""
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Handshape images must be RGBA PNGs with a transparent background")
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    opaque = image[:, :, 3] > 200
    red = opaque & ((hue < 10) | (hue > 170)) & (sat > 120) & (val > 50)
    if red.sum() > 20:
        marker = _largest_component(red)
    else:
        # Black marker: keep thick dark blobs, drop the thin outline strokes.
        dark = (opaque & (val < 60)).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        marker = _largest_component(dark)
    if marker is None:
        raise ValueError("No red or black fingertip marker found")
    return marker


def detect_anchors(image: np.ndarray) -> Dict:
    """Find the fingertip marker and the wrist in a BGRA handshape image."""
    ys, xs = np.nonzero(_marker_mask(image))
    target = [float(xs.mean()), float(ys.mean())]

    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    cuff = _largest_component((image[:, :, 3] > 200) & (sat < 40) & (val > 200))
    if cuff is None:
        raise ValueError("No white cuff found to locate the wrist")
    ys, xs = np.nonzero(cuff)
    tops = [ys[xs == x].min() for x in np.unique(xs)]
    wrist = [float(xs.mean()), float(np.median(tops))]
    return {"target": target, "wrist": wrist}


def _remove_marker(image: np.ndarray) -> np.ndarray:
    """Fill the fingertip marker with the colour of the nail around it."""
    marker = _marker_mask(image).astype(np.uint8)
    fill = cv2.dilate(marker, np.ones((5, 5), np.uint8)) > 0
    ring = (cv2.dilate(marker, np.ones((15, 15), np.uint8)) > 0) & ~fill
    ring &= (image[:, :, 3] > 200) & (cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)[:, :, 2] > 150)
    if not ring.any():
        return image
    out = image.copy()
    out[fill, :3] = np.median(image[ring][:, :3], axis=0).astype(np.uint8)
    return out


class HandShape:
    def __init__(self, image: np.ndarray, target, wrist, axis_angle: float = 90.0):
        bgra = image.astype(np.float32) / 255.0
        alpha = bgra[:, :, 3:4]
        # Premultiplied colours, so warping and blending shapes stay correct.
        self.premultiplied = np.concatenate([bgra[:, :, :3] * alpha, alpha], axis=2)
        self.target = np.array(target, dtype=np.float64)
        self.wrist = np.array(wrist, dtype=np.float64)
        self.axis_angle = axis_angle
        h, w = image.shape[:2]
        self.corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64)


class HandShapeSet:
    """The eight handshapes, ready to be composited onto video frames."""

    def __init__(self, shapes: Dict[int, HandShape]):
        self.shapes = shapes
        tips = [np.linalg.norm(s.target - s.wrist) for s in shapes.values()]
        # One reference length for all shapes, so the hand keeps its size when the shape changes.
        self.knuckle_length = float(np.median(tips)) * KNUCKLE_TO_TIP_RATIO

    @classmethod
    def load(cls, folder: str) -> "HandShapeSet":
        meta = {}
        meta_path = os.path.join(folder, METADATA_FILE)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        shapes = {}
        for n in SHAPES:
            path = os.path.join(folder, f"handshape_{n}.png")
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(f"Handshape image not found: {path}")
            anchors = meta.get(str(n)) or detect_anchors(image)
            image = _remove_marker(image)
            shapes[n] = HandShape(image, anchors["target"], anchors["wrist"],
                                  anchors.get("axis_angle", 90.0))
        return cls(shapes)

    def scale_for_face(self, face_height: float, ratio: float) -> float:
        return ratio * face_height / self.knuckle_length

    def composite(self, frame: np.ndarray, shape_weights: Dict[int, float], target,
                  angle: float, scale: float, opacity: float = 1.0) -> None:
        """Draw the hand in place so the reference fingertip lands on ``target``.

        ``shape_weights`` blends several shapes (aligned on the fingertip)
        during a shape change. ``angle`` is the direction wrist -> knuckle
        in degrees (90 = upright).
        """
        if opacity <= 0.01:
            return
        target = np.asarray(target, dtype=np.float64)
        fh, fw = frame.shape[:2]
        matrices = {}
        x0, y0, x1, y1 = fw, fh, 0, 0
        for n, weight in shape_weights.items():
            if weight <= 0.01:
                continue
            shape = self.shapes[n]
            m = cv2.getRotationMatrix2D(tuple(shape.target), angle - shape.axis_angle, scale)
            m[:, 2] += target - shape.target
            corners = shape.corners @ m[:, :2].T + m[:, 2]
            x0, y0 = min(x0, corners[:, 0].min()), min(y0, corners[:, 1].min())
            x1, y1 = max(x1, corners[:, 0].max()), max(y1, corners[:, 1].max())
            matrices[n] = (m, weight)
        x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        x1, y1 = min(fw, int(np.ceil(x1))), min(fh, int(np.ceil(y1)))
        if not matrices or x1 <= x0 or y1 <= y0:
            return

        total = sum(w for _, w in matrices.values())
        layer = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
        for n, (m, weight) in matrices.items():
            m = m.copy()
            m[:, 2] -= (x0, y0)
            layer += (weight / total) * cv2.warpAffine(
                self.shapes[n].premultiplied, m, (x1 - x0, y1 - y0),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        layer *= opacity
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        roi = layer[:, :, :3] * 255.0 + roi * (1.0 - layer[:, :, 3:4])
        frame[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


def write_metadata(folder: str) -> str:
    """Detect anchors for every handshape, write handshapes.json and a preview image."""
    meta, tiles = {}, []
    for n in SHAPES:
        image = cv2.imread(os.path.join(folder, f"handshape_{n}.png"), cv2.IMREAD_UNCHANGED)
        anchors = detect_anchors(image)
        meta[str(n)] = anchors
        alpha = image[:, :, 3:4] / 255.0
        tile = (image[:, :, :3] * alpha + 200 * (1 - alpha)).astype(np.uint8)
        for key, colour in (("target", (255, 0, 255)), ("wrist", (255, 128, 0))):
            p = tuple(int(round(v)) for v in anchors[key])
            cv2.circle(tile, p, 6, colour, 2)
        cv2.putText(tile, str(n), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        tiles.append(tile)
    h = max(t.shape[0] for t in tiles)
    preview = np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT,
                                            value=(255, 255, 255)) for t in tiles])
    cv2.imwrite(os.path.join(folder, "handshapes_preview.png"), preview)
    path = os.path.join(folder, METADATA_FILE)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


if __name__ == "__main__":
    default = os.path.join(os.path.dirname(__file__), "data", "handshapes")
    folder = sys.argv[1] if len(sys.argv) > 1 else default
    print(f"Wrote {write_metadata(folder)} and handshapes_preview.png")
