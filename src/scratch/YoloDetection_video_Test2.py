from ultralytics import YOLO
import cv2
import time
import os

# ----------------------------
# USER SETTINGS
# ----------------------------
MODEL_PATH = r"C:\Users\dell\YOLODetection\pythonProject1\MyYolo.pt"           # your model file
VIDEO_PATH = r"C:\Users\dell\Downloads\Video_Generation_With_Dog_And_Chickens.mp4"    # Windows example: r"C:\videos\coop.mp4"
# VIDEO_PATH = "/home/user/videos/coop.mp4"  # Linux/macOS example

CONF_THRES = 0.25
DOG_CLASS_NAME = "dog"

# Big + visible from far away
BOX_THICKNESS = 10
FONT_SCALE = 1.6
FONT_THICKNESS = 4

# Blink settings (seconds)
BLINK_PERIOD = 0.6      # total cycle time (on+off). smaller = faster blinking
BLINK_DUTY = 0.5        # fraction of period that is "ON" (0.5 = half the time)

# Output (optional): set to None if you don't want to save
OUT_VIDEO_PATH = "runs/chicken_security/dog_alert_blink.mp4"

# ----------------------------
# PREP OUTPUT FOLDER
# ----------------------------
if OUT_VIDEO_PATH is not None:
    os.makedirs(os.path.dirname(OUT_VIDEO_PATH), exist_ok=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = YOLO(MODEL_PATH)
names = model.names  # {id: "label"}

# Find dog class id safely (no guessing)
dog_ids = [cid for cid, name in names.items() if name.lower() == DOG_CLASS_NAME]
if not dog_ids:
    raise ValueError(f'Class "{DOG_CLASS_NAME}" not found in model.names: {names}')
DOG_ID = dog_ids[0]
print(f"[INFO] Dog class ID: {DOG_ID}")

# ----------------------------
# OPEN VIDEO
# ----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ----------------------------
# VIDEO WRITER (OPTIONAL)
# ----------------------------
writer = None
if OUT_VIDEO_PATH is not None:
    # mp4v works in most setups; if it fails, try 'avc1' or write .avi with 'XVID'
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps if fps > 0 else 25.0, (w, h))

# ----------------------------
# BLINK FUNCTION
# ----------------------------
def blink_on(now: float) -> bool:
    """Returns True when the blink should be ON at time 'now'."""
    phase = now % BLINK_PERIOD
    return phase < (BLINK_PERIOD * BLINK_DUTY)

# ----------------------------
# MAIN LOOP
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on the frame
    results = model.predict(frame, conf=CONF_THRES, verbose=False)

    dog_boxes = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            if cls_id == DOG_ID:
                dog_boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf)))

    dog_present = len(dog_boxes) > 0

    # Blink decision
    now = time.time()
    show_blink = dog_present and blink_on(now)

    # Draw blinking dog boxes (only when blink is ON)
    if show_blink:
        for (x1, y1, x2, y2, conf) in dog_boxes:
            # Red thick rectangle for DOG (blinking)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=BOX_THICKNESS)

            # Big warning label background
            text = f"DOG {conf:.2f} !!!"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 18)), (x1 + tw + 12, y1), (0, 0, 255), thickness=-1)
            cv2.putText(
                frame, text, (x1 + 6, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA
            )

    # Optional: always show a top banner when dog is present (even if blink is OFF)
    if dog_present:
        banner = "WARNING: DOG DETECTED!"
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(frame, (10, 10), (10 + tw + 20, 10 + th + 20), (0, 0, 255), thickness=-1)
        cv2.putText(frame, banner, (20, 10 + th + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    # Show video
    cv2.imshow("Chicken Security - Dog Alert (Press Q to quit)", frame)

    # Save video (optional)
    if writer is not None:
        writer.write(frame)

    # Quit key
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

# ----------------------------
# CLEANUP
# ----------------------------
cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()

if OUT_VIDEO_PATH is not None:
    print(f"[INFO] Saved output video to: {OUT_VIDEO_PATH}")
