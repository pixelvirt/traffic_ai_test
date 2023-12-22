import cv2
import os
import math
import time
from collections import Counter

from ultralytics import YOLO

from coco_names import coco_class_list, vehicle_class_list


OUTPUT_FOLDER = 'output2'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


model = YOLO("yolo_models/yolov8l.pt")
time_interval = 2


def detect_class(img, video_name):
    frame = model(img, conf=0.5, classes=[2, 3, 5, 7])
    detected_classes = []
    for r in frame:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            conf = math.ceil(box.conf * 100)/100
            cls = int(box.cls[0])
            detected_classes.append(cls)
            cv2.putText(img, f"{conf} {coco_class_list[cls]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    detected_classes_names = [coco_class_list[item] for item in detected_classes]
    counter = Counter(detected_classes_names)

    detected_vehicles = ["vehicle" if item in vehicle_class_list else item for item in detected_classes_names]
    vehicle_counter = Counter(detected_vehicles)
    vehicle_count = vehicle_counter["vehicle"]

    counter["vehicle"] = vehicle_count

    with open(f"data.txt", 'a') as f:
        f.write(f"{video_name}: {dict(counter)}\n")

    cv2.imwrite(f"{OUTPUT_FOLDER}/{video_name}.jpg", img)

    detected_classes = []


def job():
    img_1 = cv2.imread("output/camera-1.jpg")
    img_2 = cv2.imread("output/camera-2.jpg")
    img_3 = cv2.imread("output/camera-3.jpg")
    img_4 = cv2.imread("output/camera-4.jpg")

    with open(f"data.txt", 'a') as f:
        detect_class(img_1, "camera-1")
        detect_class(img_2, "camera-2")
        detect_class(img_3, "camera-3")
        detect_class(img_4, "camera-4")

while True:
    job()
    time.sleep(time_interval)
