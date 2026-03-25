

import argparse
import cv2
import time
from pathlib import Path

from ultralytics import YOLO

# COCO class list (abbreviated — see app.py for full list) 
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]

# Distinct color per class (HSV → BGR)
import numpy as np

def make_colors(n):
    colors = []
    for i in range(n):
        h = int(180 * i / n)
        hsv = np.array([[[h, 220, 220]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in bgr))
    return colors

COLORS = make_colors(len(COCO_CLASSES))


def detect_video_cli(args):
    #  Load model 
    print(f"[INFO] Loading {args.model}.pt …")
    model = YOLO(f"{args.model}.pt")

    # ── Open input video 
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.input}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit  = min(total, args.max_frames)

    print(f"[INFO] Video: {W}x{H} @ {fps:.1f} fps  |  Processing {limit}/{total} frames")

    # Output video writer 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

    all_counts = {}
    t_total    = 0.0

    for idx in range(limit):
        ret, frame = cap.read()
        if not ret:
            break

        #  Inference
        t0      = time.time()
        results = model(frame, verbose=False)
        t_total += time.time() - t0

        boxes = results[0].boxes

        #  Draw detections 
        for box in (boxes or []):
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            if conf < args.conf:
                continue

            cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "?"
            all_counts[cls_name] = all_counts.get(cls_name, 0) + 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS[cls_id % len(COLORS)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ly = max(y1 - 5, th + 5)
            cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 4, ly + 2), color, -1)
            cv2.putText(frame, label, (x1 + 2, ly - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Frame counter overlay
        cv2.putText(frame, f"Frame {idx+1}/{limit}", (8, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        writer.write(frame)

        if (idx + 1) % 30 == 0:
            print(f"  … {idx+1}/{limit} frames done")

    cap.release()
    writer.release()

    #  Summary
    avg_ms = (t_total / max(limit, 1)) * 1000
    print(f"\n Done!  Output → {args.output}")
    print(f"⏱  Avg inference: {avg_ms:.1f} ms/frame")
    print(f" Total detections: {sum(all_counts.values())}")
    for cls, n in sorted(all_counts.items(), key=lambda x: -x[1]):
        print(f"   • {cls}: {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Object Detection CLI")
    parser.add_argument("--input",      required=True,          help="Input video path")
    parser.add_argument("--output",     default="detected.mp4", help="Output video path")
    parser.add_argument("--model",      default="yolov8n",      help="yolov8n/s/m")
    parser.add_argument("--conf",       type=float, default=0.4,help="Confidence threshold")
    parser.add_argument("--max-frames", type=int, default=300,  help="Max frames to process")
    args = parser.parse_args()

    detect_video_cli(args)
