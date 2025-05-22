import requests
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch
import time
import threading
import queue

# ======= Konfigurasi =======
URL_STREAM = "http://192.168.1.8:80/"  # Ganti dengan alamat ESP32-CAM kamu
MODEL_PATH = "yolov8l.pt"              # Gunakan model kecil untuk kecepatan
device = "cuda" if torch.cuda.is_available() else "cpu"

CAR_CLASS_ID = 2
MOTOR_CLASS_ID = 3
LINE_Y = 280
DETECT_EVERY = 2

model = YOLO(MODEL_PATH).to(device)
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=0.5)

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

frame_queue = queue.Queue(maxsize=10)
tracker = VehicleTracker()
frame_count = 0


def capture_frames():
    global frame_queue
    bytes_data = bytes()
    stream = requests.get(URL_STREAM, stream=True, timeout=5)
    while True:
        bytes_data += stream.raw.read(1024)
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9' )
        if a != -1 and b != -1:
            jpg = bytes_data[a:b + 2]
            bytes_data = bytes_data[b + 2:]
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None and not frame_queue.full():
                frame_queue.put(frame)


def process_frames():
    global frame_queue, frame_count
    prev_time = time.time()

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame_count % DETECT_EVERY == 0:
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", device=device)[0]

                if results.boxes.id is None:
                    frame_count += 1
                    continue

                detections = sv.Detections.from_ultralytics(results)
                mask = np.isin(detections.class_id, [CAR_CLASS_ID, MOTOR_CLASS_ID])
                filtered_detections = detections[mask]

                if hasattr(detections, "tracker_id") and detections.tracker_id is not None:
                    filtered_detections.tracker_id = detections.tracker_id[mask]
                else:
                    frame_count += 1
                    continue

                tracker.update(filtered_detections)

                labels = [
                    f"{'MOBIL' if class_id == CAR_CLASS_ID else 'MOTOR'} {conf:.2f}"
                    for class_id, conf in zip(filtered_detections.class_id, filtered_detections.confidence)
                ]

                frame = box_annotator.annotate(frame, filtered_detections)
                frame = label_annotator.annotate(frame, filtered_detections, labels)

            cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 255, 255), 2)
            cv2.putText(frame, f"MOBIL: {tracker.car_count} | MOTOR: {tracker.motor_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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
        t1 = threading.Thread(target=capture_frames)
        t2 = threading.Thread(target=process_frames)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        print("Program dihentikan.")
    finally:
        cv2.destroyAllWindows()