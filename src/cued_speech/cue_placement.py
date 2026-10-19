"""Where the hand goes: vowel positions, hand angles and hand size from the face.

Positions and angles follow the "unified" models of Auto-CS (B. Bigi,
CNRS-LPL, https://auto-cuedspeech.org), which are defined on the 68-point
dlib face model. Here they are computed from MediaPipe FaceMesh landmarks
through an equivalence table. This is an independent implementation that
reuses only the published geometric rules and parameters.

Coordinates are image pixels (y pointing down). Angles are in degrees with
the usual orientation (y up): 90 means fingers pointing straight up, smaller
values tilt the fingers towards the face of a right-handed cuer.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Optional

import numpy as np

# dlib 68-point index -> MediaPipe FaceMesh index, for the points we need.
# dlib 0..16 run along the jaw from the subject's right (image left).
DLIB_TO_MEDIAPIPE: Dict[int, int] = {
    0: 127, 1: 234, 2: 93, 3: 132, 4: 58, 6: 136, 7: 150, 8: 152,
    19: 105, 24: 334, 27: 168, 33: 2, 36: 33, 48: 61, 57: 17, 60: 78,
}

# Hand angle (direction wrist -> middle-finger knuckle) at each position.
POSITION_ANGLES: Dict[str, float] = {
    "n": 45.0, "b": 63.0, "c": 57.0, "m": 60.0, "s": 70.0, "t": 50.0,
}

# Wrist -> middle-finger knuckle distance, as a ratio of the face height.
HAND_TO_FACE_RATIO = 0.45


class FaceGeometry:
    """Face points (dlib numbering) of one frame, with derived measures."""

    def __init__(self, points: Dict[int, np.ndarray], height: float):
        self.points = points
        self.height = height

    def x(self, i: int) -> float:
        return float(self.points[i][0])

    def y(self, i: int) -> float:
        return float(self.points[i][1])

    @property
    def angle(self) -> float:
        """Direction chin -> nose bridge; 90 when the head is upright."""
        dx = self.x(27) - self.x(8)
        dy = self.y(8) - self.y(27)
        return math.degrees(math.atan2(dy, dx))

    def vowel_positions(self) -> Dict[str, np.ndarray]:
        """Target point of the reference finger for every position code."""
        x, y = self.x, self.y
        pos = {
            "n": (x(0), y(8) + 4 * (y(8) - y(57))),
            "b": (x(4) + abs(x(36) - x(0)) / 3.7, y(1) + abs(y(2) - y(1)) / 1.4),
            "c": (x(8) - abs(x(8) - x(6)) / 3.2, y(8) - abs(y(8) - y(57)) / 4.2),
            "m": (x(48) - abs(x(48) - x(4)) / 5, y(60) + abs(y(60) - y(4)) * 0.8),
            "s": (x(0) - abs(x(8) - x(0)) / 2.6, y(4) - abs(y(4) - y(3)) / 2),
            "t": (x(8) - abs(x(8) - x(6)) / 3.6, y(8) + 0.85 * abs(y(8) - y(33))),
        }
        return {code: np.array(p, dtype=np.float64) for code, p in pos.items()}


class FaceTracker:
    """Turns per-frame MediaPipe landmarks into smoothed face geometry.

    Points are smoothed with an exponential moving average to remove
    landmark jitter; the face height is averaged over about one second.
    When no face is detected, the last known geometry is kept.
    """

    def __init__(self, fps: float, smoothing: float = 0.5):
        self.smoothing = smoothing
        self._points: Optional[Dict[int, np.ndarray]] = None
        self._heights = deque(maxlen=max(1, int(round(fps))))
        self.current: Optional[FaceGeometry] = None

    def update(self, landmarks, width: int, height: int) -> Optional[FaceGeometry]:
        if landmarks is None:
            return self.current
        raw = {
            d: np.array([landmarks.landmark[m].x * width, landmarks.landmark[m].y * height])
            for d, m in DLIB_TO_MEDIAPIPE.items()
        }
        if self._points is None:
            self._points = raw
        else:
            a = self.smoothing
            self._points = {d: a * self._points[d] + (1 - a) * raw[d] for d in raw}
        brows = (self._points[19] + self._points[24]) / 2.0
        self._heights.append(float(np.linalg.norm(brows - self._points[8])))
        self.current = FaceGeometry(self._points, float(np.mean(self._heights)))
        return self.current
