<div align="center">

# 🔍 Indoor Object Detection

### Real-time Object Detection powered by YOLOv8

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure)

</div>

---

## ✨ Features

- 🖼️ **Image Detection** - Upload and analyze images with bounding boxes and confidence scores
- 🎥 **Video Processing** - Detect objects in video files frame-by-frame
- 📷 **Live Webcam** - Real-time object detection from your camera
- 📱 **Mobile Friendly** - Use the "Browser Camera" mode on Streamlit Cloud to detect from your phone!
- 🏺 **Archaeology Ready** - Integrated support for custom artifact models
- 📊 **Detection Statistics** - Real-time object counts and metrics
- ⚙️ **Hardware Control** - Control sieve motors and read load-cell weights via Firebase

**Supported Objects:** 80 classes from COCO dataset including people, electronics, vehicles, and more!

---

## 🎬 Demo

> **Note:** Run the app to see real-time detections with bounding boxes!

```bash
# Quick start
python -m streamlit run app.py
```

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (optional, for live detection)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/3assem0/ML-CV.git
   cd ML-CV
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python -m streamlit run app.py
   ```

The app will automatically download required YOLOv8 weights on first run.

---

## 🚀 Usage

### Streamlit Dashboard (Recommended)

```bash
python -m streamlit run app.py
```

Then open `http://localhost:8501`

#### Dashboard Overview

**📊 Sidebar Controls:**
- **Model Size:** Automatically uses `yolov8n` for speed on cloud.
- **Confidence Threshold:** Adjust detection sensitivity (0.0 - 1.0)
- **Firebase Status:** Check connection to your hardware backend.

**📑 Tabs:**

1. **⚙️ Hardware Control**
   - Toggle DC Motors and Vibration system.
   - Read live weights from Load Cells (Areas 1-3).

2. **🖼️ Image/Video Upload**
   - Upload images (JPG, PNG) or videos (MP4).
   - View detections and side-by-side results.

3. **📷 Live Camera**
   - **CV2 Webcam:** For local PC usage (uses desktop camera).
   - **Browser Camera:** For Mobile/Cloud usage (uses browser snapshot).

### Command Line Scripts

- **Webcam:** `python src/webcam.py`
- **Image:** `python src/image_detect.py --image path/to/img.jpg`
- **Train:** `python src/train.py` (Local only)

---

## 🏗️ Project Structure

```
ML-CV/
├── app.py                      # Main Streamlit dashboard
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies (for Cloud)
├── data.yaml                   # Dataset config
├── README.md                   # This file
│
├── src/
│   ├── detector.py            # YOLO instance wrapper
│   ├── utils.py               # Drawing & stats helpers
│   ├── webcam.py              # CLI webcam script
│   ├── image_detect.py        # CLI image script
│   └── train.py               # Local training script
│
├── data/
│   └── raw/                   # Collection folder for images
│
├── models/                     # Custom weights (custom.pt)
│
└── esp32_firmware/            # Arduino code for hardware
```

---

## 🛠️ Technologies Used

- **[YOLOv8](https://github.com/ultralytics/ultralytics)** - State-of-the-art detection
- **[Streamlit](https://streamlit.io/)** - Web UI
- **[Firebase](https://firebase.google.com/)** - Real-time database for hardware
- **[OpenCV](https://opencv.org/)** - Computer Vision

---

## 📝 License
MIT License.

<div align="center">
**Made with ❤️ using YOLOv8 and Streamlit**
</div>