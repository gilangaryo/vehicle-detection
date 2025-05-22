import cv2
from ultralytics import YOLO
import supervision as sv
import MySQLdb
import torch
import numpy as np

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

# Definisi line zone untuk mendeteksi crossing
# line_zone = sv.LineZone(start=sv.Point(5, 400), end=sv.Point(1000, 500))
line_y = 600
line_zone = sv.LineZone(start=sv.Point(5, line_y), end=sv.Point(1000, line_y))
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.5)

# Class untuk tracker kendaraan
class VehicleTracker:
    def __init__(self):
        self.car_count = 0
        self.motor_count = 0
        self.tracked_objects = {}  # {id: {"position": y_center, "counted": False, "class_id": class_id}}
    
    def update(self, detections):
        # Jika tidak ada deteksi atau tidak ada tracker_id, keluar
        if len(detections) == 0 or detections.tracker_id is None:
            return
        
        # Update posisi objek yang sudah ditrack
        current_ids = set()
        
        for i, tracker_id in enumerate(detections.tracker_id):
            current_ids.add(tracker_id)
            bbox = detections.xyxy[i]
            class_id = detections.class_id[i]
            
            # Hanya tertarik pada mobil dan motor
            if class_id != CAR_CLASS_ID and class_id != MOTOR_CLASS_ID:
                continue
                
            y_bottom = bbox[3]  # posisi y bawah dari bounding box
            
            # Tambahkan tracker baru atau update yang sudah ada
            if tracker_id in self.tracked_objects:
                prev_pos = self.tracked_objects[tracker_id]["position"]
                is_counted = self.tracked_objects[tracker_id]["counted"]
                
                # Jika belum dihitung dan sekarang melintasi garis (dari atas ke bawah)
                
                # if not is_counted and prev_pos <= line_zone.start.y and y_bottom > line_zone.start.y:
                if not is_counted and prev_pos <= line_y and y_bottom > line_y:

                    self.tracked_objects[tracker_id]["counted"] = True
                    
                    if class_id == CAR_CLASS_ID:
                        self.car_count += 1
                        jenis = "Mobil"
                    else:
                        self.motor_count += 1
                        jenis = "Motor"
                    
                    # Insert ke database
                    cursor.execute("INSERT INTO kendaraan (jenis, keterangan) VALUES (%s, %s)", (jenis, "IN"))
                    db.commit()
                
                # Update posisi
                self.tracked_objects[tracker_id]["position"] = y_bottom
            else:
                # Tambahkan objek baru ke tracking
                self.tracked_objects[tracker_id] = {
                    "position": y_bottom,
                    "counted": False,
                    "class_id": class_id
                }
        
        # Optional: Bersihkan tracker untuk objek yang tidak lagi terdeteksi
        # for tracker_id in list(self.tracked_objects.keys()):
        #     if tracker_id not in current_ids:
        #         del self.tracked_objects[tracker_id]

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
        
        # Jika tidak ada hasil deteksi yang valid, lanjutkan ke frame berikutnya
        if results.boxes.id is None:
            sink.write_frame(frame)
            continue
        
        # Konversi hasil ke format supervision
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter hanya untuk mobil dan motor
        mask = (detections.class_id == CAR_CLASS_ID) | (detections.class_id == MOTOR_CLASS_ID)
        filtered_detections = detections[mask]
        
        # Pastikan tracker_id tersedia untuk deteksi yang difilter
        if hasattr(detections, "tracker_id") and detections.tracker_id is not None:
            filtered_detections.tracker_id = detections.tracker_id[mask]
        else:
            sink.write_frame(frame)
            continue
        
        # Update tracker dengan deteksi terbaru
        tracker.update(filtered_detections)
        
        # Buat labels untuk anotasi
        labels = [
            f"{'MOBIL' if class_id == CAR_CLASS_ID else 'MOTOR'} {conf:.2f}"
            for class_id, conf in zip(filtered_detections.class_id, filtered_detections.confidence)
        ]
        
        # Anotasi frame
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(annotated_frame, filtered_detections)
        annotated_frame = label_annotator.annotate(annotated_frame, filtered_detections, labels)
        annotated_frame = line_annotator.annotate(annotated_frame, line_zone)
        
        # Tambahkan counter
        counter_text = f"MOBIL: {tracker.car_count} | MOTOR: {tracker.motor_count}"
        cv2.putText(annotated_frame, counter_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Simpan frame
        sink.write_frame(annotated_frame)

# Output hasil akhir
print(f"Total Mobil: {tracker.car_count} | Total Motor: {tracker.motor_count}")
db.close()