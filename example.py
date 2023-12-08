import os
import math
from collections import Counter
import numpy as np

import cv2
from ultralytics import YOLO

from coco_names import coco_class_list, vehicle_class_list


VIDEO_PATH = "./videos/traffic_vid_2.mp4"
YOLO_MODEL_NANO = YOLO("yolo_models/yolov8n.pt")
YOLO_MODEL_SMALL = YOLO("yolo_models/yolov8s.pt")
YOLO_MODEL_LARGE = YOLO("yolo_models/yolov8l.pt")
OUTPUT_FOLDER = 'output'
CAPTURE_INTERVAL = 2
CONFIDENCE_THRESHOLD = 0.7


if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


video = cv2.VideoCapture(VIDEO_PATH)
if not video.isOpened():
    print("Cannot open video")
    exit()


# line coordinates
lx1, ly1 = 5, 640
lx2, ly2 = 260, 5


# distance from line
def distance_from_line(x, y):
    numerator = (lx2 - lx1) * (ly1 - y) - (lx1 - x) * (ly2 - ly1)
    denominator = np.sqrt((lx2 - lx1)**2 + (ly2 - ly1)**2)
    if denominator == 0:
        return 0
    return numerator / denominator


try:
    frame_count = 0
    fps = int(round(video.get(cv2.CAP_PROP_FPS)))

    while True:
        # read frame from webcam
        success, img = video.read()
        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # run YOLO model on each frame
        res =  cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)
        frames = YOLO_MODEL_NANO(res, stream=True, conf=CONFIDENCE_THRESHOLD)

        frame_count += 1
        cv2.line(img, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)

        for r in frames:
            detected_classes = []
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Calculate distance from the line
                distance = distance_from_line(center_x, center_y)

                # Determine whether the object is on the left or right
                if distance > 0:
                    position = "R"
                elif distance < 0:
                    position = "L"
                else:
                    position = "O"

                # Draw bounding box and label
                if position == "L":
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    conf = math.ceil(box.conf * 100)/100
                    cls = int(box.cls[0])
                    detected_classes.append(cls)
                    cv2.putText(img, f"{conf} {coco_class_list[cls]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Video", img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

finally:
    video.release()
    cv2.destroyAllWindows()
