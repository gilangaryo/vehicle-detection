import requests
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch
import time

# ======= Konfigurasi =======
URL_STREAM = "http://192.168.1.8:80/"  # Ganti dengan alamat ESP32-CAM kamu
MODEL_PATH = "yolov8x.pt"              # Gunakan model kecil untuk kecepatan
device = "cuda" if torch.cuda.is_available() else "cpu"

# Line virtual
CAR_CLASS_ID = 2
MOTOR_CLASS_ID = 3
LINE_Y = 280  # Posisi garis horizontal
DETECT_EVERY = 2  # Deteksi setiap N frame (untuk performa)

# ======= Load Model YOLO =======
model = YOLO(MODEL_PATH).to(device)

# Annotator dari Supervision
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=0.5)


# ======= Class Vehicle Tracker =======
class VehicleTracker:
    def __init__(self):
        self.car_count = 0
        self.motor_count = 0
        self.tracked_objects = {}

    def update(self, detections):
        if len(detections) == 0 or detections.tracker_id is None:
            return

        for i, tracker_id in enumerate(detections.tracker_id):
            class_id = detections.class_id[i]
            bbox = detections.xyxy[i]
            y_bottom = bbox[3]

            if class_id not in [CAR_CLASS_ID, MOTOR_CLASS_ID]:
                continue

            if tracker_id in self.tracked_objects:
                prev_y = self.tracked_objects[tracker_id]["position"]
                counted = self.tracked_objects[tracker_id]["counted"]

                if not counted and prev_y <= LINE_Y and y_bottom > LINE_Y:
                    self.tracked_objects[tracker_id]["counted"] = True
                    if class_id == CAR_CLASS_ID:
                        self.car_count += 1
                    else:
                        self.motor_count += 1

                self.tracked_objects[tracker_id]["position"] = y_bottom
            else:
                self.tracked_objects[tracker_id] = {
                    "position": y_bottom,
                    "counted": False,
                    "class_id": class_id
                }


# ======= Fungsi utama =======
def mjpeg_stream_count(url):
    tracker = VehicleTracker()
    frame_count = 0
    bytes_data = bytes()
    stream = requests.get(url, stream=True, timeout=5)
    prev_time = time.time()

    while True:
        bytes_data += stream.raw.read(1024)
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_data[a:b + 2]
            bytes_data = bytes_data[b + 2:]
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # frame = cv2.resize(frame, (329, 240))

            # Deteksi tiap beberapa frame
            if frame_count % DETECT_EVERY == 0:
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", device=device)[0]

                if results.boxes.id is None:
                    frame_count += 1
                    continue

                detections = sv.Detections.from_ultralytics(results)

                # Filter hanya mobil dan motor
                mask = np.isin(detections.class_id, [CAR_CLASS_ID, MOTOR_CLASS_ID])
                filtered_detections = detections[mask]

                if hasattr(detections, "tracker_id") and detections.tracker_id is not None:
                    filtered_detections.tracker_id = detections.tracker_id[mask]
                else:
                    frame_count += 1
                    continue

                # Update tracking dan counting
                tracker.update(filtered_detections)

                # Anotasi bounding box dan label
                labels = [
                    f"{'MOBIL' if class_id == CAR_CLASS_ID else 'MOTOR'} {conf:.2f}"
                    for class_id, conf in zip(filtered_detections.class_id, filtered_detections.confidence)
                ]

                frame = box_annotator.annotate(frame, filtered_detections)
                frame = label_annotator.annotate(frame, filtered_detections, labels)

            # Garis virtual
            cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 255, 255), 2)

            # Counter kendaraan
            cv2.putText(frame, f"MOBIL: {tracker.car_count} | MOTOR: {tracker.motor_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Hitung dan tampilkan FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.imshow("ESP32 Detection", frame)
            if cv2.waitKey(1) == 27:
                break

            frame_count += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        mjpeg_stream_count(URL_STREAM)
    # Jika ingin menghentikan program dengan Ctrl+C
    except KeyboardInterrupt:
        print("Program dihentikan.")
    finally:
        cv2.destroyAllWindows()