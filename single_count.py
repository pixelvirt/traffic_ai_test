import math
from collections import Counter

import cv2
from ultralytics import YOLO

from coco_names import coco_class_list, vehicle_class_list


model = YOLO("yolo_models/yolov8l.pt")


# vid_name = "traffic_vid_1.mp4"
vid_name = "both_side.mp4"


video_1 = cv2.VideoCapture(f"./videos/{vid_name}")


if not video_1.isOpened():
    print("Cannot open video")
    exit()


def detect_class(img, video_name):
    frame = model(img, conf=0.5, classes=[2, 3, 5, 7])
    for r in frame:
        cls = r.boxes.cls
        object_classes = [coco_class_list[int(item)] for item in cls]

        counter = dict(Counter(object_classes))
        vehicle_count = 0
        for key in counter:
            if key in vehicle_class_list:
                vehicle_count += counter[key]
        
        counter["vehicle"] = vehicle_count

        with open(f"data.txt", 'a') as f:
            f.write(f"{video_name}: {counter}\n")


try:
    frame_count = 0
    fps = int(round(video_1.get(cv2.CAP_PROP_FPS)))
    time_interval = 1

    while video_1.isOpened():
        success_1, img_1 = video_1.read()

        frame_count += 1
        
        if not success_1:
            break

        if frame_count % (fps * time_interval) == 0:
            detect_class(img_1, "camera-1")


finally:
    video_1.release()
    cv2.destroyAllWindows()