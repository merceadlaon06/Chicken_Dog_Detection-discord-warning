from ultralytics import YOLO
import cv2
import numpy as np
import os
import urllib.request

# ----------------------------
# USER SETTINGS
# ----------------------------
MODEL_PATH = r"C:\Users\dell\YOLODetection\pythonProject1\MyYolo.pt"
IMAGE_URL = "https://smartchickendoor.b-cdn.net/wp-content/uploads/Blog/COMPLETE-Predator-Blogs/Dog-Blog/dogs-and-chickens-as-toys-1.jpg"  # <-- put your URL here

CONF_THRES = 0.25
DOG_CLASS_NAME = "dog"

# Visible from far away
BOX_THICKNESS = 10
FONT_SCALE = 1.5
FONT_THICKNESS = 4

# Output
OUT_DIR = "runs/chicken_security"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_IMAGE = os.path.join(OUT_DIR, "detect_image_result.jpg")


# ----------------------------
# HELPER: DOWNLOAD IMAGE FROM URL
# ----------------------------
def load_image_from_url(url: str):
    """
    Downloads an image from a URL and returns it as an OpenCV BGR image.
    """
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    img_array = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Failed to decode image from URL. Check if the URL is a direct image link.")
    return img


# ----------------------------
# LOAD MODEL
# ----------------------------
model = YOLO(MODEL_PATH)

# ----------------------------
# RUN DETECTION ON URL
# ----------------------------
results = model.predict(
    source=IMAGE_URL,     # YOLO can accept URL directly
    conf=CONF_THRES,
    verbose=False
)

# Class names dict: {id: "label"}
names = model.names

# Find DOG class id safely
dog_class_ids = [cid for cid, name in names.items() if name.lower() == DOG_CLASS_NAME]
if not dog_class_ids:
    raise ValueError(f'Class "{DOG_CLASS_NAME}" not found in model.names: {names}')
DOG_ID = dog_class_ids[0]

print(f"Dog class ID: {DOG_ID}")

# ----------------------------
# LOAD IMAGE (FOR DRAWING)
# ----------------------------
img = load_image_from_url(IMAGE_URL)

dog_detected = False

# ----------------------------
# PROCESS RESULTS + DRAW
# ----------------------------
for r in results:
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        continue

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss  = boxes.cls.cpu().numpy().astype(int)

    # Print raw outputs like your sample
    print(boxes.xyxy)
    print(boxes.conf)
    print(boxes.cls)

    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
        label = names.get(cls_id, str(cls_id))
        is_dog = (cls_id == DOG_ID)

        if is_dog:
            dog_detected = True

        # Red for dog, green for others
        color = (0, 0, 255) if is_dog else (0, 255, 0)

        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # Thick bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=BOX_THICKNESS)

        # Label with background (easy to read far away)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)

        cv2.rectangle(img, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, thickness=-1)
        cv2.putText(
            img, text,
            (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
            cv2.LINE_AA
        )

print("names:", names)

# ----------------------------
# ALERT
# ----------------------------
if dog_detected:
    print("⚠️ WARNING: DOG DETECTED! Chickens may be in danger!")
else:
    print("✅ No dog detected.")

# ----------------------------
# SAVE OUTPUT
# ----------------------------
cv2.imwrite(OUT_IMAGE, img)
print(f"Saved annotated image to: {OUT_IMAGE}")
