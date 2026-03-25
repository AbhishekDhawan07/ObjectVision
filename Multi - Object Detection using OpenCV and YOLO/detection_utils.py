"""
utils/detection_utils.py
────────────────────────
Standalone Computer Vision utilities for Multi-Object Detection.
These functions are imported by app.py but can also be used independently
in notebooks, scripts, or other backends.

Topics covered:
  • Frame-level preprocessing with OpenCV
  • Post-processing YOLO results (NMS already done inside ultralytics)
  • Tracking object counts across a video timeline
  • Saving annotated frames as a timelapse / summary grid
"""

import cv2
import numpy as np
from pathlib import Path
import json
import time
from typing import Optional



# A. PREPROCESSING UTILITIES


def preprocess_frame(
    frame: np.ndarray,
    target_size: Optional[tuple] = None,
    normalize: bool = False,
) -> np.ndarray:
    """
    Preprocess a BGR frame before inference.

    Steps performed:
      1. Optionally resize while preserving aspect ratio (letterbox).
      2. Optionally normalize pixel values to [0, 1].

    Args:
        frame      : Input BGR image as NumPy array
        target_size: (width, height) tuple, or None to skip resize
        normalize  : If True, convert to float32 and divide by 255

    Returns:
        Processed frame
    """
    processed = frame.copy()

    if target_size is not None:
        # Letterbox resize: pad with gray to keep aspect ratio
        processed = letterbox_resize(processed, target_size)

    if normalize:
        # Convert to float32 in [0,1] — used for custom inference pipelines
        processed = processed.astype(np.float32) / 255.0

    return processed


def letterbox_resize(
    img: np.ndarray,
    target_size: tuple,
    pad_color: tuple = (114, 114, 114)
) -> np.ndarray:
    """
    Resize image to target_size using letterboxing (no distortion).
    Adds gray padding on shorter dimension.

    Args:
        img        : BGR NumPy image
        target_size: (width, height)
        pad_color  : BGR padding color (default: gray 114,114,114 — YOLO standard)

    Returns:
        Resized + padded BGR image
    """
    tw, th = target_size
    ih, iw = img.shape[:2]

    # Compute scale so image fits inside target without cropping
    scale = min(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    # Resize with high-quality INTER_LINEAR interpolation
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Create canvas and paste resized image centered
    canvas = np.full((th, tw, 3), pad_color, dtype=np.uint8)
    x_off = (tw - nw) // 2
    y_off = (th - nh) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized

    return canvas



# B. POST-PROCESSING UTILITIES


def filter_detections(
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
    confidences: np.ndarray,
    class_names: list,
    conf_threshold: float = 0.4,
    allowed_classes: Optional[list] = None,
) -> list[dict]:
    """
    Filter raw YOLO detections by confidence and class name.

    Args:
        boxes_xyxy    : (N, 4) array of [x1,y1,x2,y2] in pixel coords
        class_ids     : (N,) array of integer class indices
        confidences   : (N,) array of float confidence scores
        class_names   : List of COCO class name strings
        conf_threshold: Minimum confidence to keep a detection
        allowed_classes: Whitelist of class name strings (None = all)

    Returns:
        List of dicts: {box, class_id, class_name, confidence}
    """
    detections = []
    for i, (box, cls_id, conf) in enumerate(zip(boxes_xyxy, class_ids, confidences)):
        if conf < conf_threshold:
            continue

        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

        if allowed_classes and cls_name not in allowed_classes:
            continue

        detections.append({
            "box"       : box.tolist(),   # [x1, y1, x2, y2]
            "class_id"  : int(cls_id),
            "class_name": cls_name,
            "confidence": float(conf),
        })

    return detections


def compute_iou(box_a: list, box_b: list) -> float:
    """
    Compute Intersection-over-Union between two [x1,y1,x2,y2] boxes.
    Useful for custom NMS implementations or tracking.

    Args:
        box_a, box_b: Lists [x1, y1, x2, y2]

    Returns:
        IoU score in [0, 1]
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0



# C. VIDEO ANALYTICS — TIMELINE TRACKER

class DetectionTimeline:
    """
    Records per-frame detection counts to generate analytics
    (object counts over time, peak frames, class distributions).

    Usage:
        tracker = DetectionTimeline()
        for frame_idx, counts in frame_detections:
            tracker.record(frame_idx, counts)
        summary = tracker.summary()
    """

    def __init__(self):
        self._records: list[dict] = []   # [{frame, counts, timestamp}]

    def record(self, frame_idx: int, counts: dict, timestamp: Optional[float] = None):
        """
        Store detection counts for one frame.

        Args:
            frame_idx : 0-based frame index
            counts    : {class_name: count} dict
            timestamp : Optional wall-clock time (seconds)
        """
        self._records.append({
            "frame"    : frame_idx,
            "counts"   : counts.copy(),
            "timestamp": timestamp or time.time(),
        })

    def total_detections(self) -> dict:
        """Sum detections across all frames by class."""
        totals = {}
        for rec in self._records:
            for cls, n in rec["counts"].items():
                totals[cls] = totals.get(cls, 0) + n
        return totals

    def peak_frame(self) -> dict:
        """Return the frame with most total detections."""
        if not self._records:
            return {}
        return max(self._records, key=lambda r: sum(r["counts"].values()))

    def avg_per_class(self) -> dict:
        """Average detections per frame for each class."""
        if not self._records:
            return {}
        totals = self.total_detections()
        n = len(self._records)
        return {cls: round(count / n, 2) for cls, count in totals.items()}

    def summary(self) -> dict:
        """Return complete analytics dict."""
        return {
            "total_frames"    : len(self._records),
            "total_detections": self.total_detections(),
            "peak_frame"      : self.peak_frame(),
            "avg_per_class"   : self.avg_per_class(),
        }

    def export_json(self, path: str):
        """Save timeline records to a JSON file for offline analysis."""
        with open(path, "w") as f:
            json.dump(self._records, f, indent=2)
        print(f"[Timeline] Saved {len(self._records)} records → {path}")



# D. SUMMARY GRID — VISUAL SAMPLE SHEET

def build_summary_grid(
    frames: list[np.ndarray],
    cols: int = 4,
    cell_size: tuple = (320, 180),
) -> np.ndarray:
    """
    Arrange a list of BGR frames into a grid image (contact sheet).
    Useful for quick visual QA of video detections.

    Args:
        frames   : List of BGR NumPy frames
        cols     : Number of columns in the grid
        cell_size: (width, height) of each thumbnail

    Returns:
        Single BGR image containing all thumbnails in a grid
    """
    cw, ch = cell_size
    rows = int(np.ceil(len(frames) / cols))

    # Black canvas
    grid = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        thumb = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_AREA)

        y0, y1 = row * ch, (row + 1) * ch
        x0, x1 = col * cw, (col + 1) * cw
        grid[y0:y1, x0:x1] = thumb

        # Frame number label
        cv2.putText(
            grid, f"#{idx}",
            (x0 + 4, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (200, 200, 200), 1, cv2.LINE_AA
        )

    return grid



# E. OPTICAL FLOW — MOTION HEATMAP (BONUS)

def compute_motion_heatmap(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
) -> np.ndarray:
    """
    Compute a dense optical-flow motion heatmap between two frames.
    Useful for highlighting fast-moving objects (cars, bikes).

    Uses Farneback dense optical flow (cv2.calcOpticalFlowFarneback).

    Args:
        frame_prev: Previous BGR frame
        frame_curr: Current BGR frame

    Returns:
        BGR heatmap image (same size as input)
    """
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)

    # Farneback dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray_prev, gray_curr,
        None,
        pyr_scale=0.5,   # image pyramid scale
        levels=3,        # pyramid levels
        winsize=15,      # averaging window size
        iterations=3,
        poly_n=5,        # pixel neighborhood size
        poly_sigma=1.2,  # Gaussian std for polynomial expansion
        flags=0,
    )

    # Compute magnitude and angle of flow vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Map to HSV: hue=direction, saturation=1, value=magnitude
    hsv = np.zeros_like(frame_prev)
    hsv[..., 0] = angle * 180 / np.pi / 2   # hue (direction)
    hsv[..., 1] = 255                        # full saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    heatmap_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return heatmap_bgr
