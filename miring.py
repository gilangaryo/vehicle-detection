import cv2
from ultralytics import YOLO
import supervision as sv
import MySQLdb
import torch
import numpy as np
from supervision.geometry.core import Point

# Inisialisasi koneksi database
db = MySQLdb.connect(host="localhost", user="root", passwd="", db="parkir")
cursor = db.cursor()

# Konfigurasi
SOURCE_VIDEO_PATH = "./sample/sample1.mp4"
TARGET_VIDEO_PATH = "output.mp4"
MODEL = "yolov8l.pt"
CAR_CLASS_ID = 2
MOTOR_CLASS_ID = 3

# Inisialisasi model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Menggunakan: {device}")
model = YOLO(MODEL).to(device)

# Definisi line zone untuk garis miring (contoh: dari kiri bawah ke kanan atas)
line_start = Point(100, 500)  # Titik awal (x1, y1)
line_end = Point(900, 300)    # Titik akhir (x2, y2)
line_zone = sv.LineZone(start=line_start, end=line_end)
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.5)

# Class untuk tracker kendaraan
class VehicleTracker:
    def __init__(self):
        self.car_count = 0
        self.motor_count = 0
        self.tracked_objects = {}  # {id: {"position": Point, "counted": False, "class_id": class_id}}

    @staticmethod
    def is_crossing_line(prev_point, curr_point, line_start, line_end):
        def side(p, a, b):
            return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
        return side(prev_point, line_start, line_end) * side(curr_point, line_start, line_end) < 0

    def update(self, detections):
        if len(detections) == 0 or detections.tracker_id is None:
            return

        current_ids = set()

        for i, tracker_id in enumerate(detections.tracker_id):
            current_ids.add(tracker_id)
            bbox = detections.xyxy[i]
            class_id = detections.class_id[i]

            if class_id != CAR_CLASS_ID and class_id != MOTOR_CLASS_ID:
                continue

            curr_center = Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            if tracker_id in self.tracked_objects:
                prev_center = self.tracked_objects[tracker_id]["position"]
                is_counted = self.tracked_objects[tracker_id]["counted"]

                if not is_counted and VehicleTracker.is_crossing_line(prev_center, curr_center, line_start, line_end):
                    self.tracked_objects[tracker_id]["counted"] = True

                    if class_id == CAR_CLASS_ID:
                        self.car_count += 1
                        jenis = "Mobil"
                    else:
                        self.motor_count += 1
                        jenis = "Motor"

                    # Simpan ke database
                    cursor.execute("INSERT INTO kendaraan (jenis, keterangan) VALUES (%s, %s)", (jenis, "IN"))
                    db.commit()

                self.tracked_objects[tracker_id]["position"] = curr_center
            else:
                self.tracked_objects[tracker_id] = {
                    "position": curr_center,
                    "counted": False,
                    "class_id": class_id
                }

# Inisialisasi vehicle tracker
tracker = VehicleTracker()

# Set up anotasi
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=0.5)

with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame in sv.get_video_frames_generator(SOURCE_VIDEO_PATH):
        # Deteksi dan tracking objek
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", device=device)[0]
        
        if results.boxes.id is None:
            sink.write_frame(frame)
            continue
        
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter hanya mobil dan motor
        mask = (detections.class_id == CAR_CLASS_ID) | (detections.class_id == MOTOR_CLASS_ID)
        filtered_detections = detections[mask]
        
        if hasattr(detections, "tracker_id") and detections.tracker_id is not None:
            filtered_detections.tracker_id = detections.tracker_id[mask]
        else:
            sink.write_frame(frame)
            continue
        
        tracker.update(filtered_detections)
        
        labels = [
            f"{'MOBIL' if class_id == CAR_CLASS_ID else 'MOTOR'} {conf:.2f}"
            for class_id, conf in zip(filtered_detections.class_id, filtered_detections.confidence)
        ]
        
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(annotated_frame, filtered_detections)
        annotated_frame = label_annotator.annotate(annotated_frame, filtered_detections, labels)
        annotated_frame = line_annotator.annotate(annotated_frame, line_zone)
        
        counter_text = f"MOBIL: {tracker.car_count} | MOTOR: {tracker.motor_count}"
        cv2.putText(annotated_frame, counter_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        sink.write_frame(annotated_frame)

print(f"Total Mobil: {tracker.car_count} | Total Motor: {tracker.motor_count}")
db.close()