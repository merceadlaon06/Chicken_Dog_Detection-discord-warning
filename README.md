# 🐔 Chicken Security System

## Real-Time Dog Detection + Discord Alerting using YOLOv8

An end-to-end computer vision security system that detects **dogs** in video streams using a custom-trained YOLOv8 model and triggers real-time **visual alerts + Discord notifications with image snapshots**.

---

## 🎥 Demo (Testing Output)

![Dog Alert Demo](test_output/dog_alert_blink.gif)

> 📌 *Sample output showing real-time detection, blinking alerts, and system response.*

---

## 🚀 Project Overview

This system continuously monitors a video feed and automatically:

- Detects dogs using a YOLOv8 model
- Displays high-visibility blinking alerts
- Sends Discord notifications with timestamps
- Attaches a snapshot image of the detection
- Prevents alert spamming using cooldown logic
- Optionally saves an annotated output video

This project demonstrates a **production-style computer vision inference pipeline** adaptable to CCTV systems, local video feeds, or RTSP streams.

---

## 🧠 Key Features

### 🔍 Intelligent Detection

- YOLOv8-based object detection
- Confidence threshold filtering
- Dynamic class ID resolution (no hardcoded indices)

### 🚨 Operator-Friendly Visual Alerts

- Thick red bounding boxes
- Large warning labels with confidence scores
- Blinking effect for enhanced visibility
- Top banner alert when a dog is detected

### 📢 Smart Discord Notification System

- Webhook-based integration
- Timestamped alert messages
- Snapshot image attachments
- Cooldown mechanism to prevent spam
- Edge-trigger logic (alerts only on new detection events)

### 💾 Optional Video Export

Annotated output video is saved to:

```
runs/chicken_security/dog_alert_blink.mp4
```

---

## 🏗️ System Architecture

### Pipeline Flow

1. Load YOLO model (`MyYolo.pt`)
2. Capture frames using OpenCV
3. Perform object detection inference
4. Filter detections for the target class ("dog")
5. Render blinking overlays and alerts
6. Trigger Discord notifications (with cooldown protection)
7. Save annotated output video (optional)

---

## 🛠 Tech Stack

- Python 3.9+
- Ultralytics YOLOv8
- OpenCV
- Requests (Discord Webhook API)

---

## 📦 Installation

```bash
pip install ultralytics opencv-python requests
```

---

## ⚙️ Configuration

Modify the following variables in your script:

- `MODEL_PATH`
- `VIDEO_PATH`
- `CONF_THRES`
- `DOG_CLASS_NAME`
- `DISCORD_WEBHOOK_URL`
- `DISCORD_COOLDOWN_SEC`
- `SEND_FRAME_IMAGE`
- `OUT_VIDEO_PATH`

---

## ▶️ Usage

```bash
python main.py
```

Press **Q** to exit the program.

---

## 🔮 Future Improvements

- RTSP live stream support
- Object tracking (DeepSORT / ByteTrack)
- Region-of-interest (ROI) based alerts
- Logging and monitoring system
- CLI argument support
- Docker containerization

---

## 🎯 Why This Project Matters

This project showcases:

- End-to-end computer vision pipeline design
- Event-driven alert architecture
- Production-ready notification handling
- Real-world deployment considerations
- Model-agnostic system design

---

## 👤 Author

**Your Name**  
GitHub: https://github.com/YOUR_USERNAME  
LinkedIn: https://linkedin.com/in/YOUR_PROFILE

---

## 📜 License

MIT License
