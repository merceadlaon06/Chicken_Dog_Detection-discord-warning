import cv2
import time
import os
import requests
from datetime import datetime
from ultralytics import YOLO

# ----------------------------
# CONFIGURATION
# ----------------------------
# Using your specific directory path
OUTPUT_DIR = r"C:\Users\dell\YOLODetection\pythonProject1\test_output"

CONFIG = {
    "model_path": r"C:\Users\dell\YOLODetection\pythonProject1\MyYolo.pt",
    "video_path": r"C:\Users\dell\Downloads\Merce Downloads- 2_15_2026\Video_Generation_With_Dog_And_Chickens.mp4",
    # We will build the full filename in the Class constructor
    "output_dir": OUTPUT_DIR,
    "conf_threshold": 0.25,
    "target_class": "dog",
    "discord_webhook": "Add your webhook URL",
    "discord_cooldown": 30,
    "blink_period": 0.6,
    "blink_duty": 0.5
}


class DogSecuritySystem:
    def __init__(self, config):
        self.cfg = config
        self.model = YOLO(self.cfg["model_path"])
        self.dog_id = self._get_class_id(self.cfg["target_class"])

        # State Management
        self.last_discord_sent = 0.0
        self.was_dog_present = False

        # ----------------------------
        # OUTPUT DIRECTORY SETUP
        # ----------------------------
        if not os.path.exists(self.cfg["output_dir"]):
            os.makedirs(self.cfg["output_dir"], exist_ok=True)
            print(f"[INFO] Created directory: {self.cfg['output_dir']}")

        # Create a unique filename based on current time to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.full_output_path = os.path.join(self.cfg["output_dir"], f"alert_log_{timestamp}.mp4")

    def _get_class_id(self, class_name):
        names = self.model.names
        ids = [cid for cid, name in names.items() if name.lower() == class_name.lower()]
        if not ids:
            raise ValueError(f"Class '{class_name}' not found in model.")
        return ids[0]

    def is_blink_on(self) -> bool:
        return (time.time() % self.cfg["blink_period"]) < (self.cfg["blink_period"] * self.cfg["blink_duty"])

    def send_discord_alert(self, message: str, frame):
        if not self.cfg["discord_webhook"] or "webhook" in self.cfg["discord_webhook"]:
            return
        try:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                files = {"file": ("alert.jpg", buf.tobytes(), "image/jpeg")}
                requests.post(self.cfg["discord_webhook"], data={"content": message}, files=files, timeout=8)
        except Exception as e:
            print(f"[ERROR] Discord failed: {e}")

    def process_detections(self, results):
        dog_boxes = []
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                if int(box.cls[0]) == self.dog_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    dog_boxes.append((x1, y1, x2, y2, conf))
        return dog_boxes

    def draw_overlays(self, frame, dog_boxes):
        dog_present = len(dog_boxes) > 0
        if dog_present:
            cv2.putText(frame, "WARNING: DOG DETECTED!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if dog_present and self.is_blink_on():
            for (x1, y1, x2, y2, conf) in dog_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
                cv2.putText(frame, f"DOG {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    def run(self):
        cap = cv2.VideoCapture(self.cfg["video_path"])
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {self.cfg['video_path']}")
            return

        # Setup Video Writer with the new path
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(self.full_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        print(f"[INFO] Security Monitor active. Saving to: {self.full_output_path}")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                results = self.model.predict(frame, conf=self.cfg["conf_threshold"], verbose=False)
                dog_boxes = self.process_detections(results)

                # Discord Alert Trigger
                if dog_boxes:
                    now = time.time()
                    if not self.was_dog_present or (now - self.last_discord_sent > self.cfg["discord_cooldown"]):
                        self.send_discord_alert(f"🚨 Dog detected at {datetime.now().strftime('%H:%M:%S')}!", frame)
                        self.last_discord_sent = now
                    self.was_dog_present = True
                else:
                    self.was_dog_present = False

                self.draw_overlays(frame, dog_boxes)
                cv2.imshow("Security Feed", frame)
                writer.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            print(f"[INFO] Finished. Video saved at: {self.full_output_path}")


if __name__ == "__main__":
    app = DogSecuritySystem(CONFIG)
    app.run()