import os
import math
from collections import Counter

import cv2
from ultralytics import YOLO

from coco_names import coco_class_list, vehicle_class_list


VIDEO_PATH = "./videos/traffic_1.mp4"
# VIDEO_PATH = "./videos/traffic_vid_1.mp4"
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
        # frames = YOLO_MODEL_NANO(img, stream=True, conf=CONFIDENCE_THRESHOLD)
        # frames = YOLO_MODEL_SMALL(img, stream=True, conf=CONFIDENCE_THRESHOLD)
        frames = YOLO_MODEL_LARGE(img, stream=True, conf=CONFIDENCE_THRESHOLD)

        frame_count += 1

        for r in frames:
            detected_classes = []
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                conf = math.ceil(box.conf * 100)/100
                cls = int(box.cls[0])
                detected_classes.append(cls)
                cv2.putText(img, f"{conf} {coco_class_list[cls]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # save frame to output folder
            if frame_count % (CAPTURE_INTERVAL * fps) == 0:
                detected_classes_names = [coco_class_list[item] for item in detected_classes]
                counter = Counter(detected_classes_names)
                # print(counter)

                detected_vehicles = ["vehicle" if item in vehicle_class_list else item for item in detected_classes_names]
                vehicle_counter = Counter(detected_vehicles)
                vehicle_count = vehicle_counter["vehicle"]

                counter["vehicle"] = vehicle_count

                with open(f"data.txt", 'a') as f:
                    f.write(f"{dict(counter)}\n")

                text_x, text_y = 25, 25
                for class_name, count in counter.items():
                    cv2.putText(img, f"{class_name}: {count}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    text_y += 25

                cv2.imwrite(f"{OUTPUT_FOLDER}/{frame_count}.jpg", img)
                # print(f"Captured {frame_count}.jpg")
                
            detected_classes = []

        cv2.imshow("Video", img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

finally:
    video.release()
    cv2.destroyAllWindows()
