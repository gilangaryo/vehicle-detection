import cv2
from ultralytics import YOLO
import supervision as sv
import MySQLdb
import torch

# MySQL connection
db = MySQLdb.connect(host="localhost", user="root", passwd="", db="parkir")
cursor = db.cursor()

# Config
SOURCE_STREAM_URL = "http://192.168.1.8/"  # Use your ESP32 IP with trailing slash
MODEL_PATH = "yolov8l.pt"
CAR_CLASS_ID = 2
MOTOR_CLASS_ID = 3

# Load model and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = YOLO(MODEL_PATH).to(device)

# Line for counting (y coordinate)
line_y = 230
line_zone = sv.LineZone(start=sv.Point(5, line_y), end=sv.Point(1000, line_y))

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

            y_bottom = bbox[3]  # bottom y coordinate of bbox
            if tracker_id in self.tracked_objects:
                prev_pos = self.tracked_objects[tracker_id]["position"]
                is_counted = self.tracked_objects[tracker_id]["counted"]
                if not is_counted and prev_pos <= line_y < y_bottom:
                    self.tracked_objects[tracker_id]["counted"] = True
                    jenis = "Mobil" if class_id == CAR_CLASS_ID else "Motor"
                    if class_id == CAR_CLASS_ID:
                        self.car_count += 1
                    else:
                        self.motor_count += 1
                    # Insert record to MySQL
                    cursor.execute("INSERT INTO kendaraan (jenis, keterangan) VALUES (%s, %s)", (jenis, "IN"))
                    db.commit()
                    print(f"{jenis} counted. Total mobil: {self.car_count}, motor: {self.motor_count}")
                self.tracked_objects[tracker_id]["position"] = y_bottom
            else:
                self.tracked_objects[tracker_id] = {
                    "position": y_bottom,
                    "counted": False,
                    "class_id": class_id
                }

tracker = VehicleTracker()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=0.5)

cap = cv2.VideoCapture(SOURCE_STREAM_URL)
if not cap.isOpened():
    print("Failed to open video stream")
    exit()

print("Stream opened successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", device=device)[0]

    detections = sv.Detections.from_ultralytics(results)

    # Filter only cars and motorbikes
    mask = (detections.class_id == CAR_CLASS_ID) | (detections.class_id == MOTOR_CLASS_ID)
    filtered_detections = detections[mask]

    if hasattr(detections, "tracker_id") and detections.tracker_id is not None:
        filtered_detections.tracker_id = detections.tracker_id[mask]
    else:
        continue

    tracker.update(filtered_detections)

    labels = [
        f"{'MOBIL' if class_id == CAR_CLASS_ID else 'MOTOR'} {conf:.2f}"
        for class_id, conf in zip(filtered_detections.class_id, filtered_detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(annotated_frame, filtered_detections)
    annotated_frame = label_annotator.annotate(annotated_frame, filtered_detections, labels)

    # Draw counting line
    cv2.line(annotated_frame, (0, line_y), (annotated_frame.shape[1], line_y), (0, 0, 255), 2)

    counter_text = f"MOBIL: {tracker.car_count} | MOTOR: {tracker.motor_count}"
    cv2.putText(annotated_frame, counter_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("ESP32 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
db.close()

print(f"Total Mobil: {tracker.car_count} | Total Motor: {tracker.motor_count}")
