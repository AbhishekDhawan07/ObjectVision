# 🎯✨Multi-Object Detection using OpenCV and YOLO ✨🎯

<div align="center">

```
 ██████╗ ██████╗      ██╗███████╗ ██████╗████████╗    ██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗
██╔═══██╗██╔══██╗     ██║██╔════╝██╔════╝╚══██╔══╝    ██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║
██║   ██║██████╔╝     ██║█████╗  ██║        ██║       ██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║
██║   ██║██╔══██╗██   ██║██╔══╝  ██║        ██║       ╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║
╚██████╔╝██████╔╝╚█████╔╝███████╗╚██████╗   ██║        ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║
 ╚═════╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝         ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
```

### 🚗💨 Real-Time Multi-Object Detection · YOLOv8 + OpenCV + Gradio 🎨🖥️

<br/>

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/🤖_YOLOv8-Ultralytics-FF6600?style=for-the-badge)
![Gradio](https://img.shields.io/badge/🎨_Gradio-Web_UI-FF4B4B?style=for-the-badge)
![FFmpeg](https://img.shields.io/badge/FFmpeg-H.264-007808?style=for-the-badge&logo=ffmpeg&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-F7DF1E?style=for-the-badge)

<br/>

> 🎯 **Detect 80 COCO classes in videos** · ⚡ **Real-time inference** · 🖥️ **Web UI + CLI** · 📊 **Analytics built-in**

<br/>

[🎬 View Demo](#-demo) &nbsp;·&nbsp; [🚀 Quick Start](#-getting-started) &nbsp;·&nbsp; [📖 Docs](#️-how-it-works) &nbsp;·&nbsp; [🐛 Report Bug](https://github.com/ObjectVision/Multi---Object-Detection-using-OpenCV-and-YOLO/issues) &nbsp;·&nbsp; [💡 Request Feature](https://github.com/ObjectVision/Multi---Object-Detection-using-OpenCV-and-YOLO/issues)

</div>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Demo](#-demo)
- [Tech Stack](#️-tech-stack)
- [Features](#-features)
- [Dataset](#-dataset)
- [How It Works](#️-how-it-works)
- [Project Structure](#️-project-structure)
- [Getting Started](#-getting-started)
- [Contributing](#-contributing)
- [License](#️-license)

---

## 🧠💡 About the Project

**🎯 Multi-Object Detection using OpenCV and YOLO** is a production-ready computer vision pipeline that detects and tracks **80 COCO object classes** across video streams — in real time, frame by frame. 🎥

Built inside the 🏗️ **[ObjectVision](https://github.com/ObjectVision)** repository, this project fuses:

- 🤖 The blazing speed of **YOLOv8** for state-of-the-art object detection
- 🔬 The power of **OpenCV** for frame preprocessing, annotation & optical flow
- 🖥️ An interactive **Gradio** dark-themed web dashboard for zero-code use
- ⌨️ A **CLI runner** for headless batch processing on servers or pipelines

Whether you're building a 🚦 traffic monitor, 🔒 surveillance system, 🏎️ vehicle tracker, or just exploring modern CV — this project gives you a **clean, modular, and extensible** foundation to build on.

> 📌 **Repository:** [ObjectVision / Multi - Object Detection using OpenCV and YOLO](https://github.com/ObjectVision/Multi---Object-Detection-using-OpenCV-and-YOLO)

---

## 🎬🍿 Demo

### 🔍✨ Detection Output Video

The video below is the fully annotated output — YOLOv8 running inference on real-world traffic footage, with colored bounding boxes, class labels, and confidence scores rendered on every single frame:

<div align="center">

### 📥 [`detected.mp4`](./Multi%20-%20Object%20Detection%20using%20OpenCV%20and%20YOLO/detected.mp4) — 🎯 Annotated Output Video
</div>

> 🟢 **Colored boxes** = detected objects &nbsp;|&nbsp; 🏷️ **Labels** = class name + confidence score &nbsp;|&nbsp; 🔢 **Counter** = frame progress overlay

### 🎞️ What You'll See in the Demo

| 🎭 Detection Element | 📝 Description |
|:---:|:---|
| 🟥🟦🟩 **Colored Bounding Boxes** | Each class gets a unique HSV-derived color for instant visual separation |
| 🏷️ **Label Pills** | Filled background tag showing class name + confidence (e.g. `car 0.87`) |
| 🔢 **Frame Counter** | Live `Frame X / N` overlay at the bottom of every frame |
| 📊 **Stats Panel** | Inference time, total detections, and per-class breakdown in the UI |

---

## 🛠️⚙️ Tech Stack

<div align="center">

| 🔩 Layer | 🚀 Technology | 💬 Role |
|:---:|:---:|:---|
| 🤖 **Detection Engine** | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | State-of-the-art object detection (nano / small / medium) |
| 👁️ **Computer Vision** | [OpenCV 4.x](https://opencv.org/) | Frame I/O, drawing, letterbox resize, optical flow |
| 🖥️ **Web Interface** | [Gradio](https://gradio.app/) | Interactive dark-themed video detection dashboard |
| 🎞️ **Video Encoding** | FFmpeg + H.264 | Browser-compatible MP4 with `yuv420p` + `faststart` |
| 🐍 **Language** | Python 3.8+ | Core runtime |
| 🧠 **Model Weights** | `yolov8n.pt` | COCO-pretrained weights (80 classes, included in repo) |
| 🔢 **Numerics** | NumPy | Array ops, HSV color generation, flow magnitude |

</div>

---

## ✨🌟 Features

### 🎯 Core Detection
- 🏷️ **80-class COCO detection** — people 🧑, cars 🚗, bikes 🚲, buses 🚌, animals 🐕, and 75 more!
- 🎚️ **Configurable confidence threshold** — filter weak detections, keep only strong ones
- 🎨 **Per-class color coding** — visually distinct HSV-derived bounding box colors
- 🔍 **Class whitelist filter** — detect *only* the objects you care about

### 🖥️ Interfaces
- 🌐 **Gradio Web UI** — upload video, tune settings, watch results — **zero code needed**
- ⌨️ **CLI Runner** (`run_on_video.py`) — headless batch processing with full `argparse` support
- 🔧 **Utility Library** (`detection_utils.py`) — clean, reusable, importable CV functions

### 📊 Analytics & Intelligence
- 📈 **Detection statistics** — per-class counts, avg inference time ⏱️, total detections 🎯
- 🗂️ **DetectionTimeline tracker** — records counts across frames for temporal analytics
- 🖼️ **Summary grid builder** — contact sheet of sampled annotated frames for visual QA
- 🌊 **Optical flow / motion heatmap** — Farneback dense flow for 💨 motion analysis

### 🏗️ Engineering Quality
- 📐 **Letterbox preprocessing** — distortion-free resizing before inference
- 🔄 **FFmpeg re-encoding** — guaranteed browser-compatible MP4 with `faststart` streaming
- ⚡ **Model caching** — singleton loader avoids redundant disk reads on every call

---

## 📦🎞️ Dataset

The input video is a real-world urban traffic scene captured from dashcam footage:

<div align="center">

### 📥 [`cars_youtube.mp4`](./Multi%20-%20Object%20Detection%20using%20OpenCV%20and%20YOLO/cars_youtube.mp4) — 🚦 Input Dataset Video

</div>

| 🏷️ Property | 📋 Value |
|:---:|:---|
| 📡 **Source** | YouTube traffic / dashcam footage |
| 🌆 **Content** | Urban road scene — cars, motorcycles, buses, pedestrians |
| 🎞️ **Format** | MP4 (H.264) |
| 🏷️ **COCO Classes Present** | `🚗 car` · `🚛 truck` · `🚌 bus` · `🏍️ motorcycle` · `🧑 person` · `🚲 bicycle` |

> 🧠 The YOLO model is pretrained on the full **COCO 2017** dataset — **80 classes, 118k training images**. No additional training required — pure inference mode. 🎉

---

## 🔄⚙️ How It Works

```
🎥 Input Video (MP4)
        │
        ▼
┌──────────────────────────────────┐
│ 📂  Frame Extraction             │
│     cv2.VideoCapture             │
│     → BGR frames at native FPS   │
└─────────────┬────────────────────┘
              │ 🖼️ raw BGR frames
              ▼
┌──────────────────────────────────┐
│ 🔧  Preprocessing                │
│     • 📐 Letterbox resize        │
│     • 🔢 Optional normalization  │
└─────────────┬────────────────────┘
              │ ✅ preprocessed frame
              ▼
┌──────────────────────────────────┐
│ 🤖  YOLOv8 Inference             │
│     model(frame, verbose=False)  │
│     → 📦 boxes, 🏷️ class IDs,   │
│        📊 confidence scores      │
└─────────────┬────────────────────┘
              │ 🎯 raw detections
              ▼
┌──────────────────────────────────┐
│ 🔍  Post-Processing              │
│     • 🎚️ Confidence filter       │
│     • 🏷️ Class whitelist filter  │
│     • 📐 IoU / NMS (built-in)    │
└─────────────┬────────────────────┘
              │ ✅ filtered detections
              ▼
┌──────────────────────────────────┐
│ 🎨  OpenCV Annotation            │
│     • 🟩 Colored bounding boxes  │
│     • 🏷️ Label + confidence pill │
│     • 🔢 Frame counter overlay   │
└─────────────┬────────────────────┘
              │ 🖼️ annotated BGR frame
              ▼
┌──────────────────────────────────┐
│ 💾  Video Writer + Re-Encoder    │
│     cv2.VideoWriter → raw MP4    │
│     🔄 FFmpeg → H.264 + faststart│
└─────────────┬────────────────────┘
              │
              ▼
   🎬 Output Video + 📊 Stats
```

### 🗂️ Key Modules at a Glance

| 📄 File | 🎯 Role | 🔑 Key Functions |
|:---|:---:|:---|
| 🖥️ `OpenCVgradioapp_video.py` | Gradio Web App | `draw_detections()` · `detect_video()` · `load_model()` |
| ⌨️ `run_on_video.py` | CLI Runner | `detect_video_cli()` · argparse entrypoint |
| 🔧 `detection_utils.py` | Utility Library | `preprocess_frame()` · `letterbox_resize()` · `filter_detections()` · `compute_iou()` · `DetectionTimeline` · `build_summary_grid()` · `compute_motion_heatmap()` |

---

## 📁🗂️ Project Structure

```
🏗️ ObjectVision/
├── 📄  README.md                          ← 📖 You are here!
└── 📂  Multi - Object Detection using OpenCV and YOLO/
    │
    ├── 🖥️  OpenCVgradioapp_video.py       ← Gradio web app (main entry point)
    ├── ⌨️  run_on_video.py                ← CLI runner for headless processing
    ├── 🔧  detection_utils.py             ← Reusable CV utility library
    │
    ├── 🤖  yolov8n.pt                     ← YOLOv8 nano pretrained weights (COCO)
    │
    ├── 📹  cars_youtube.mp4               ← 🎞️ Input dataset video (traffic footage)
    ├── 🎬  detected.mp4                   ← ✅ Auto-generated output (created after running the app)
    │
```

---

## 🚀💻 Getting Started

### 🛒 Prerequisites

| 🔩 Requirement | 📋 Details |
|:---:|:---|
| 🐍 Python | **3.8 or higher** |
| 📦 pip | Latest version recommended |
| 🎞️ FFmpeg | *(Optional but highly recommended)* for browser-compatible H.264 output |

### 📥 Step 1 — Clone the Repository

```bash
# 🌀 Clone the repo
git clone https://github.com/ObjectVision/Multi---Object-Detection-using-OpenCV-and-YOLO.git

# 📂 Navigate into the project folder
cd "Multi - Object Detection using OpenCV and YOLO"
```

### 📦 Step 2 — Install Dependencies

```bash
# 🔧 Install all required packages in one shot
pip install ultralytics opencv-python gradio numpy Pillow
```

> 💡 **Pro Tip:** The `yolov8n.pt` weights are already bundled in the repo! Larger models (`yolov8s`, `yolov8m`) will be auto-downloaded by ultralytics on first use. 🤖✨

### 🖥️ Step 3a — Launch the Gradio Web App

```bash
# 🚀 Fire up the web interface
python OpenCVgradioapp_video.py
```

🌐 Open your browser at **`http://localhost:7862`** and follow these steps:

| 🔢 Step | 🎯 Action |
|:---:|:---|
| 1️⃣ | 📹 **Upload** your video file |
| 2️⃣ | 🤖 **Choose** model size — `nano` ⚡ / `small` ⚖️ / `medium` 🎯 |
| 3️⃣ | 🎚️ **Adjust** confidence threshold & class filter |
| 4️⃣ | ⏩ **Set** max frames to process |
| 5️⃣ | ▶️ **Click** "Process Video" → get annotated output + 📊 stats! |

### ⌨️ Step 3b — Run the CLI Script

```bash
# 🎬 Run detection on a video via command line
python run_on_video.py \
  --input      cars_youtube.mp4 \
  --output     detected.mp4 \
  --model      yolov8n \
  --conf       0.4 \
  --max-frames 300
```

#### 🗂️ CLI Arguments Reference

| 🏷️ Argument | ⚙️ Default | 📋 Description |
|:---:|:---:|:---|
| `--input` | *(required)* 📌 | 📹 Path to input video file |
| `--output` | `detected.mp4` 🎬 | 💾 Output video file path |
| `--model` | `yolov8n` 🤖 | 🧠 Size: `yolov8n` ⚡ / `yolov8s` ⚖️ / `yolov8m` 🎯 |
| `--conf` | `0.4` 🎚️ | 📊 Confidence threshold (0.1 – 0.95) |
| `--max-frames` | `300` ⏩ | 🔢 Maximum frames to process |

### 🔧 Step 4 — Use the Utility Library in Your Own Code

```python
# 📦 Import only what you need
from detection_utils import (
    preprocess_frame,        # 🔧 Letterbox resize + normalize
    filter_detections,       # 🔍 Filter by class & confidence
    DetectionTimeline,       # 📈 Track counts across frames
    build_summary_grid,      # 🖼️ Create contact sheet of frames
    compute_motion_heatmap,  # 🌊 Dense optical flow heatmap
)

# 📐 Letterbox-resize a frame before custom inference
processed = preprocess_frame(frame, target_size=(640, 640))
print("✅ Frame preprocessed:", processed.shape)

# 📈 Track detection counts across frames
timeline = DetectionTimeline()
timeline.record(frame_idx=0, counts={"car": 3, "person": 1})
timeline.record(frame_idx=1, counts={"car": 4, "motorcycle": 2})

# 📊 Get full analytics summary
print(timeline.summary())
# → {'total_frames': 2, 'total_detections': {...}, 'peak_frame': {...}, ...}
```

---

## 🤝🌍 Contributing

🎉 Contributions are what make open-source such an amazing place to **learn**, **inspire**, and **create**! Any contributions you make are **greatly appreciated**. 💪🚀

### 🔀 How to Contribute

```
1. 🍴  Fork the Project
2. 🌿  Create your Feature Branch  →  git checkout -b feature/AmazingFeature
3. 💾  Commit your Changes         →  git commit -m '✨ Add some AmazingFeature'
4. 📤  Push to the Branch          →  git push origin feature/AmazingFeature
5. 🔁  Open a Pull Request         →  and we'll review it! 🎉
```

### 💡 Ideas & Roadmap

| 🏷️ Priority | 💡 Feature Idea |
|:---:|:---|
| 🔥 High | 📡 Real-time webcam / RTSP stream detection tab in Gradio |
| 🔥 High | 🏃 Object tracking with **ByteTrack** / **DeepSORT** |
| ⭐ Medium | 🤖 Support for `yolov8l` and `yolov8x` larger models |
| ⭐ Medium | 📊 Export detection timeline to CSV / JSON for offline analysis |
| 💡 Nice-to-have | 🖼️ Gradio single-frame image detection tab |
| 💡 Nice-to-have | 🐳 Docker container for one-command deployment |
| 💡 Nice-to-have | 📱 Mobile-friendly Gradio responsive layout |
| 💡 Nice-to-have | 🌊 Live optical flow overlay toggle in the web UI |

---

## ⚖️📄 License

Distributed under the **MIT License** 🟡 — free to use, modify, and distribute with attribution.
See [`LICENSE`](./LICENSE) for full details.

---

<div align="center">

---

### 🌟 If this project helped you, please drop a ⭐ star — it means the world! 🙏💙

---

🎯 Built with 💙 &nbsp;·&nbsp; 🤖 Powered by YOLOv8 &nbsp;·&nbsp; 👁️ Visualized with OpenCV &nbsp;·&nbsp; 🖥️ Served by Gradio

**🏗️ Part of the [ObjectVision](https://github.com/ObjectVision) Repository**

<br/>

[![GitHub Stars](https://img.shields.io/github/stars/ObjectVision/Multi---Object-Detection-using-OpenCV-and-YOLO?style=social)](https://github.com/ObjectVision/Multi---Object-Detection-using-OpenCV-and-YOLO/stargazers)
&nbsp;&nbsp;
[![GitHub Forks](https://img.shields.io/github/forks/ObjectVision/Multi---Object-Detection-using-OpenCV-and-YOLO?style=social)](https://github.com/ObjectVision/Multi---Object-Detection-using-OpenCV-and-YOLO/network/members)
&nbsp;&nbsp;
[![GitHub Issues](https://img.shields.io/github/issues/ObjectVision/Multi---Object-Detection-using-OpenCV-and-YOLO?style=social)](https://github.com/ObjectVision/Multi---Object-Detection-using-OpenCV-and-YOLO/issues)

</div>

