import cv2
import numpy as np
import gradio as gr
import tempfile
import os
from pathlib import Path
from PIL import Image
import time


# 1. IMPORTS & MODEL SETUP


try:
    from ultralytics import YOLO  # YOLOv8 via ultralytics library
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Run: pip install ultralytics")


# 2. COCO CLASS LABELS (80 classes)


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


# 3. GENERATE DISTINCT COLORS PER CLASS
#    Using HSV color space for visual clarity


def generate_colors(num_classes: int) -> list:
    """Generate visually distinct BGR colors for each class."""
    colors = []
    for i in range(num_classes):
        hue = int(180 * i / num_classes)          # spread hues evenly
        color_hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color_bgr))
    return colors

CLASS_COLORS = generate_colors(len(COCO_CLASSES))

# 4. MODEL LOADER (cached singleton)


_model_cache = {}

def load_model(model_size: str = "yolov8n") -> "YOLO":
    """
    Load a YOLOv8 model. Uses an in-memory cache to avoid
    reloading on every inference call.

    Args:
        model_size: One of 'yolov8n', 'yolov8s', 'yolov8m'
    Returns:
        Loaded YOLO model instance
    """
    if model_size not in _model_cache:
        print(f"[INFO] Loading model: {model_size}.pt ...")
        _model_cache[model_size] = YOLO(f"{model_size}.pt")  # auto-downloads
        print(f"[INFO] Model '{model_size}' loaded successfully.")
    return _model_cache[model_size]


# 5. CORE DETECTION FUNCTION (OpenCV + YOLO)


def draw_detections(
    frame: np.ndarray,
    results,
    conf_threshold: float,
    selected_classes: list,
    show_conf: bool,
    show_labels: bool,
    box_thickness: int,
) -> tuple[np.ndarray, dict]:
    """
    Draw bounding boxes and labels on a frame using OpenCV.

    Args:
        frame         : BGR NumPy image
        results       : YOLOv8 Results object
        conf_threshold: Minimum confidence to draw a box
        selected_classes: List of class names to detect (filter)
        show_conf     : Whether to display confidence score
        show_labels   : Whether to display class labels
        box_thickness : Bounding box line thickness

    Returns:
        annotated_frame : Drawn BGR image
        detection_counts: Dict of {class_name: count}
    """
    annotated = frame.copy()
    detection_counts = {}

    # Extract boxes from YOLOv8 result (first image in batch)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return annotated, detection_counts

    for box in boxes:
        # Confidence & class filtering 
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        if conf < conf_threshold:
            continue

        cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"

        if selected_classes and cls_name not in selected_classes:
            continue

        #  Count detections per class 
        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1

        #  Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

        # Draw rectangle using OpenCV
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

        #  Label construction 
        label_parts = []
        if show_labels:
            label_parts.append(cls_name)
        if show_conf:
            label_parts.append(f"{conf:.2f}")
        label = " ".join(label_parts)

        if label:
            # Background pill for readability
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            label_y = max(y1 - 5, th + 5)
            cv2.rectangle(
                annotated,
                (x1, label_y - th - baseline - 2),
                (x1 + tw + 4, label_y + baseline - 2),
                color, -1  # filled rectangle
            )
            cv2.putText(
                annotated, label,
                (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255),  # white text
                1, cv2.LINE_AA
            )

    return annotated, detection_counts


# 6. VIDEO INFERENCE

def detect_video(
    video_path: str,
    model_size: str,
    conf_threshold: float,
    selected_classes: list,
    show_conf: bool,
    show_labels: bool,
    box_thickness: int,
    max_frames: int = 300,   # cap to avoid OOM on HF Spaces
) -> tuple:
    """
    Run YOLO inference on every frame of a video.
    Writes annotated frames to a new MP4 file.

    Returns:
        (output_video_path, stats_text)
    """
    if video_path is None:
        return None, "No video provided."

    if not YOLO_AVAILABLE:
        return None, "❌ ultralytics not installed."

    #  Open video with OpenCV 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "❌ Could not open video file."

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write to a temp raw file first, then re-encode with ffmpeg for browser compat
    tmp_raw = tempfile.NamedTemporaryFile(suffix="_raw.mp4", delete=False).name
    out_path = tempfile.NamedTemporaryFile(suffix="_detected.mp4", delete=False).name

    # Try H.264 first (browser-compatible), fall back to mp4v
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(tmp_raw, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_raw, fourcc, fps, (width, height))

    model = load_model(model_size)

    all_counts  = {}
    frame_idx   = 0
    total_time  = 0.0
    process_limit = min(total_frames, max_frames)

    print(f"[INFO] Processing {process_limit}/{total_frames} frames at {fps:.1f} FPS ...")

    while frame_idx < process_limit:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        t0 = time.time()
        results = model(frame_bgr, verbose=False)
        total_time += time.time() - t0

        annotated_bgr, counts = draw_detections(
            frame_bgr, results, conf_threshold,
            selected_classes, show_conf, show_labels, box_thickness
        )

        # Overlay frame counter on video
        cv2.putText(
            annotated_bgr,
            f"Frame {frame_idx+1}/{process_limit}",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
        )

        writer.write(annotated_bgr)

        # Accumulate class counts
        for cls, n in counts.items():
            all_counts[cls] = all_counts.get(cls, 0) + n

        frame_idx += 1

    cap.release()
    writer.release()

    # Re-encode to H.264/AAC MP4 using ffmpeg for guaranteed browser playback
    ffmpeg_ok = False
    try:
        import subprocess
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", tmp_raw,
                "-vcodec", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",   # required for browser compat
                "-movflags", "+faststart",  # enables streaming playback
                out_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300
        )
        ffmpeg_ok = result.returncode == 0
        if not ffmpeg_ok:
            print(f"[WARN] ffmpeg failed: {result.stderr.decode()[:300]}")
    except Exception as e:
        print(f"[WARN] ffmpeg not available: {e}")

    # If ffmpeg failed, fall back to the raw file
    final_path = out_path if ffmpeg_ok else tmp_raw
    if ffmpeg_ok and os.path.exists(tmp_raw):
        os.remove(tmp_raw)

    avg_ms    = (total_time / max(frame_idx, 1)) * 1000
    total_det = sum(all_counts.values())

    encode_note = "" if ffmpeg_ok else "\n⚠️ ffmpeg unavailable — video may not play in browser. Install ffmpeg for best results."

    stats  = f"✅ Processed {frame_idx} frames\n"
    stats += f"⏱ Avg inference: {avg_ms:.1f} ms/frame\n"
    stats += f"🎯 Total detections: {total_det}\n"
    stats += encode_note + "\n"
    if all_counts:
        stats += "\n".join([f"  • {cls}: {n}" for cls, n in sorted(all_counts.items())])
    else:
        stats += "  No objects detected above threshold."

    return final_path, stats


# 8. GRADIO UI DEFINITION


#  Shared controls 

def build_controls():
    model_size = gr.Dropdown(
        choices=["yolov8n", "yolov8s", "yolov8m"],
        value="yolov8n",
        label="🤖 YOLO Model",
        info="nano=fastest, small=balanced, medium=accurate"
    )
    conf_slider = gr.Slider(
        minimum=0.1, maximum=0.95, value=0.40, step=0.05,
        label="🎚 Confidence Threshold"
    )
    class_filter = gr.Dropdown(
        choices=COCO_CLASSES,
        multiselect=True,
        value=[],
        label="🏷 Filter Classes (empty = all)"
    )
    show_conf   = gr.Checkbox(value=True,  label="Show Confidence Score")
    show_labels = gr.Checkbox(value=True,  label="Show Class Labels")
    box_thick   = gr.Slider(minimum=1, maximum=6, value=2, step=1, label="Box Thickness")
    return model_size, conf_slider, class_filter, show_conf, show_labels, box_thick


#  Theme 

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
).set(
    body_background_fill="#0f172a",
    block_background_fill="#1e293b",
    block_label_background_fill="#334155",
    input_background_fill="#1e293b",
    body_text_color="#e2e8f0",
    block_title_text_color="#93c5fd",
)


# 9. GRADIO BLOCKS LAYOUT


with gr.Blocks(theme=theme, title="🎬 Video Object Detection | YOLOv8") as demo:

    # Header
    gr.Markdown("""
    # 🎬 Video Moving Object Detection with YOLOv8 + OpenCV
    **Detect 80 COCO classes in videos in real-time.**
    Powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) · Built with Gradio
    """)

    # VIDEO DETECTION
    with gr.Row():
        with gr.Column(scale=1):
            vid_input = gr.Video(label="Upload Video (MP4 recommended)")
            m2, c2, cf2, sc2, sl2, bt2 = build_controls()
            max_frames_slider = gr.Slider(
                minimum=30, maximum=500, value=150, step=30,
                label="⏩ Max Frames to Process",
                info="Limit frames to avoid long waits"
            )
            vid_btn = gr.Button("▶ Process Video", variant="primary", size="lg")

        with gr.Column(scale=1):
            vid_output = gr.Video(label="Detected Output Video")
            vid_stats  = gr.Textbox(
                label="📊 Detection Stats",
                lines=12, interactive=False
            )

    vid_btn.click(
        fn=detect_video,
        inputs=[vid_input, m2, c2, cf2, sc2, sl2, bt2, max_frames_slider],
        outputs=[vid_output, vid_stats]
    )
   


# 10. LAUNCH


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # required for HuggingFace Spaces
        server_port=7862,
        share=True,
    )
