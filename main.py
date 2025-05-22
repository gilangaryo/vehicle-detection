import requests
import time
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch
import MySQLdb
import osw

# === Konfigurasi ===
MODEL_PATH = "yolov8m.pt"
CAR_CLASS_ID = 2
MOTOR_CLASS_ID = 3
LINE_Y = 440
STREAM_URL = "https://cctvjss.jogjakota.go.id/atcs/ATCS_Kleringan_Abu_Bakar_Ali.stream/chunklist_w1262374255.m3u8"

# === Inisialisasi Database ===
db = MySQLdb.connect(host="localhost", user="root", passwd="", db="parkir")
cursor = db.cursor()

# === Inisialisasi YOLO ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Menggunakan: {device}")
model = YOLO(MODEL_PATH).to(device)

# === Line Deteksi ===
line_zone = sv.LineZone(start=sv.Point(0, LINE_Y), end=sv.Point(5280, LINE_Y))
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.5)

# === Tracker Kendaraan ===
class VehicleTracker:
    def __init__(self):
        self.car_count = 0
        self.motor_count = 0
        self.tracked_objects = {}

    def update(self, detections):
        if len(detections) == 0 or detections.tracker_id is None:
            return

        for i, tracker_id in enumerate(detections.tracker_id):
            bbox = detections.xyxy[i]
            class_id = detections.class_id[i]

            if class_id != CAR_CLASS_ID and class_id != MOTOR_CLASS_ID:
                continue

            y_bottom = bbox[3]

            if tracker_id in self.tracked_objects:
                prev_pos = self.tracked_objects[tracker_id]["position"]
                is_counted = self.tracked_objects[tracker_id]["counted"]

                if not is_counted and prev_pos <= LINE_Y and y_bottom > LINE_Y:
                    self.tracked_objects[tracker_id]["counted"] = True

                    if class_id == CAR_CLASS_ID:
                        self.car_count += 1
                        jenis = "Mobil"
                    else:
                        self.motor_count += 1
                        jenis = "Motor"

                    cursor.execute("INSERT INTO kendaraan (jenis, keterangan) VALUES (%s, %s)", (jenis, "IN"))
                    db.commit()

                self.tracked_objects[tracker_id]["position"] = y_bottom
            else:
                self.tracked_objects[tracker_id] = {
                    "position": y_bottom,
                    "counted": False,
                    "class_id": class_id
                }

# === Inisialisasi Tracker dan Annotator ===
tracker = VehicleTracker()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=0.5)

# === Loop utama: download dan proses video tiap 10 detik ===
segment_index = 32  # ganti sesuai segmen awal
cap = cv2.VideoCapture(STREAM_URL)

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame, mencoba ulang...")
            time.sleep(2)
            continue

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", device=device)[0]

        if results.boxes.id is None:
            continue

        detections = sv.Detections.from_ultralytics(results)
        mask = (detections.class_id == CAR_CLASS_ID) | (detections.class_id == MOTOR_CLASS_ID)
        filtered = detections[mask]

        if hasattr(detections, "tracker_id") and detections.tracker_id is not None:
            filtered.tracker_id = detections.tracker_id[mask]
        else:
            continue

        tracker.update(filtered)

        labels = [
            f"{'MOBIL' if class_id == CAR_CLASS_ID else 'MOTOR'} {conf:.2f}"
            for class_id, conf in zip(filtered.class_id, filtered.confidence)
        ]

        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(annotated_frame, filtered)
        annotated_frame = label_annotator.annotate(annotated_frame, filtered, labels)
        annotated_frame = line_annotator.annotate(annotated_frame, line_zone)
        cv2.putText(
            annotated_frame,
            f"MOBIL: {tracker.car_count} | MOTOR: {tracker.motor_count}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Deteksi CCTV", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("Dihentikan oleh user.")
        break
    except Exception as e:
        print(f"[!] Error: {e}")
        time.sleep(2)

cap.release()
cv2.destroyAllWindows()
db.close()

print(f"Total Mobil: {tracker.car_count} | Total Motor: {tracker.motor_count}")
