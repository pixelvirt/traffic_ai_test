import cv2
import os
import math
import time
from collections import Counter

import schedule

from ultralytics import YOLO

from coco_names import coco_class_list, vehicle_class_list


OUTPUT_FOLDER = 'output_2'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


model = YOLO("yolo_models/yolov8x.pt")
time_interval = 1


def detect_class(img, video_name):
    res =  cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)
    frame = model(res, conf=0.5)
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
        f.write(f"start-{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")

        detect_class(img_1, "camera-1")
        detect_class(img_2, "camera-2")
        detect_class(img_3, "camera-3")
        detect_class(img_4, "camera-4")

        f.write(f"end-{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")


schedule.every(time_interval).seconds.do(job)

while True:
    schedule.run_pending()