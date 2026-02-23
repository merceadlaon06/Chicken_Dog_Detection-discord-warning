# 🐔 Chicken Security System

## Real-Time Dog Detection + Discord Alerting using YOLOv8

An end-to-end computer vision security system that detects **dogs** in
video streams using a custom-trained YOLO model and triggers real-time
**visual warnings + Discord notifications with image snapshots**.

------------------------------------------------------------------------

## 🚀 Project Overview

This system monitors a video feed and automatically:

-   Detects dogs using a YOLO model
-   Displays high-visibility blinking alerts
-   Sends Discord notifications with timestamps
-   Attaches a frame snapshot of the detection
-   Prevents alert spamming using cooldown logic
-   Optionally saves an annotated output video

This project demonstrates a production-style CV inference pipeline
adaptable to CCTV or RTSP streams.

------------------------------------------------------------------------

## 🧠 Key Features

### 🔍 Intelligent Detection

-   YOLOv8-based inference
-   Confidence threshold filtering
-   Dynamic class ID resolution (no hardcoded indices)

### 🚨 Operator-Friendly Visual Alerts

-   Thick red bounding boxes
-   Large warning label with confidence score
-   Blinking effect for visibility
-   Top banner alert when a dog is present

### 📢 Smart Discord Notification System

-   Webhook integration
-   Timestamped alert messages
-   Snapshot image attachment
-   Cooldown protection to prevent spam
-   Edge-trigger logic (alerts on new detection events)

### 💾 Optional Video Export

Output saved to:

    runs/chicken_security/dog_alert_blink.mp4

------------------------------------------------------------------------

## 🏗️ System Architecture

Pipeline:

1.  Load YOLO model (`MyYolo.pt`)
2.  Read frames using OpenCV
3.  Run inference
4.  Filter detections to target class ("dog")
5.  Draw blinking overlays
6.  Send Discord alerts (cooldown protected)
7.  Save annotated output video

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python 3.9+
-   Ultralytics YOLO (YOLOv8)
-   OpenCV
-   Requests (Discord Webhook API)

------------------------------------------------------------------------

## 📦 Installation

``` bash
pip install ultralytics opencv-python requests
```

------------------------------------------------------------------------

## ⚙️ Configuration

Edit these variables in the script:

-   `MODEL_PATH`
-   `VIDEO_PATH`
-   `CONF_THRES`
-   `DOG_CLASS_NAME`
-   `DISCORD_WEBHOOK_URL`
-   `DISCORD_COOLDOWN_SEC`
-   `SEND_FRAME_IMAGE`
-   `OUT_VIDEO_PATH`

------------------------------------------------------------------------

## ▶️ Usage

``` bash
python main.py
```

Press **Q** to quit.

------------------------------------------------------------------------

## 🔮 Future Improvements

-   RTSP live stream support
-   Object tracking (DeepSORT / ByteTrack)
-   ROI-based alerts
-   Logging system
-   CLI argument support
-   Docker deployment

------------------------------------------------------------------------

## 🎯 Why This Project Matters

This project demonstrates:

-   End-to-end CV pipeline implementation
-   Event-driven alert architecture
-   Production-aware notification logic
-   Real-world deployment thinking
-   Model-agnostic design decisions

------------------------------------------------------------------------

## 👤 Author

Your Name\
GitHub: https://github.com/YOUR_USERNAME\
LinkedIn: https://linkedin.com/in/YOUR_PROFILE

------------------------------------------------------------------------

## 📜 License

MIT License
