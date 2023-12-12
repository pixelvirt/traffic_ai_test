import os
import math 
from collections import Counter
import threading

import cv2
from ultralytics import YOLO

from coco_names import coco_class_list, vehicle_class_list


VIDEO_PATH_1 = "./videos/traffic_vid_1.mp4"
VIDEO_PATH_2 = "./videos/traffic_vid_1.mp4"
VIDEO_PATH_2 = "./videos/traffic_vid_1.mp4"
VIDEO_PATH_4 = "./videos/traffic_vid_1.mp4"


YOLO_MODEL_NANO = YOLO("yolo_models/yolov8n.pt")
YOLO_MODEL_LARGE = YOLO("yolo_models/yolov8l.pt")

OUTPUT_FOLDER = 'output'
CAPTURE_INTERVAL = 3
CONFIDENCE_THRESHOLD = 0.4


if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


video_1 = cv2.VideoCapture(VIDEO_PATH_1)
if not video_1.isOpened():
    print("Cannot open video")
    exit()

video_2 = cv2.VideoCapture(VIDEO_PATH_2)
if not video_2.isOpened():
    print("Cannot open video")
    exit()


video_3 = cv2.VideoCapture(VIDEO_PATH_2)
if not video_2.isOpened():
    print("Cannot open video")
    exit()


video_4 = cv2.VideoCapture(VIDEO_PATH_2)
if not video_2.isOpened():
    print("Cannot open video")
    exit()


def draw_rectangles(img):
    frame = YOLO_MODEL_NANO(img, stream=True, conf=CONFIDENCE_THRESHOLD)
    for r in frame:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            # conf = math.ceil(box.conf * 100)/100
            cls = int(box.cls[0])
            cv2.putText(img, f"{coco_class_list[cls]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def detect_objects(image, video_name):
    res =  cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)

    frame = YOLO_MODEL_LARGE(res, stream=True, conf=CONFIDENCE_THRESHOLD)
    detected_classes = []
    for r in frame:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            conf = math.ceil(box.conf * 100)/100
            cls = int(box.cls[0])
            detected_classes.append(cls)
            cv2.putText(image, f"{conf} {coco_class_list[cls]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    detected_classes_names = [coco_class_list[item] for item in detected_classes]
    counter = Counter(detected_classes_names)
    # print(counter)

    detected_vehicles = ["vehicle" if item in vehicle_class_list else item for item in detected_classes_names]
    vehicle_counter = Counter(detected_vehicles)
    vehicle_count = vehicle_counter["vehicle"]

    counter["vehicle"] = vehicle_count

    with open(f"data.txt", 'a') as f:
        f.write(f"{video_name}: {dict(counter)}\n")

    text_x, text_y = 25, 25
    for class_name, count in counter.items():
        cv2.putText(image, f"{class_name}: {count}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
        text_y += 25

    cv2.imwrite(f"{OUTPUT_FOLDER}/{frame_count}-{video_name}.jpg", image)
    # print(f"Captured {frame_count}.jpg")

    detected_classes = []


try:
    frame_count = 0
    fps = int(round(video_1.get(cv2.CAP_PROP_FPS)))

    while video_1.isOpened() and video_2.isOpened() and video_3.isOpened() and video_4.isOpened():
        # read frame from webcam
        success_1, img_1 = video_1.read()
        success_2, img_2 = video_2.read()
        success_3, img_3 = video_3.read()
        success_4, img_4 = video_4.read()

        img_1 = cv2.resize(img_1, (640, 480))
        img_2 = cv2.resize(img_2, (640, 480))
        img_3 = cv2.resize(img_3, (640, 480))
        img_4 = cv2.resize(img_4, (640, 480))

        if not success_1 or not success_2 or not success_3 or not success_4:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_count += 1

        if frame_count % 3 == 0:
            thread_1 = threading.Thread(target=draw_rectangles, args=(img_1,))
            thread_2 = threading.Thread(target=draw_rectangles, args=(img_2,))
            thread_3 = threading.Thread(target=draw_rectangles, args=(img_3,))
            thread_4 = threading.Thread(target=draw_rectangles, args=(img_4,))

            thread_1.start(), thread_2.start(), thread_3.start(), thread_4.start()

            thread_1.join(), thread_2.join(), thread_3.join(), thread_4.join()

            frame_1 = cv2.hconcat([img_1, img_2])
            frame_2 = cv2.hconcat([img_3, img_4])
            frame = cv2.vconcat([frame_1, frame_2])

            cv2.imshow('frame', frame)

        if frame_count % (CAPTURE_INTERVAL * fps) == 0:
            thread_1 = threading.Thread(target=detect_objects, args=(img_1.copy(), "1"))
            thread_2 = threading.Thread(target=detect_objects, args=(img_2.copy(), "2"))
            thread_3 = threading.Thread(target=detect_objects, args=(img_3.copy(), "3"))
            thread_4 = threading.Thread(target=detect_objects, args=(img_4.copy(), "4"))

            thread_1.start(), thread_2.start(), thread_3.start(), thread_4.start()

            # thread_1.join(), thread_2.join(), thread_3.join(), thread_4.join()

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    video_1.release()
    video_2.release()
    video_3.release()
    video_4.release()
    cv2.destroyAllWindows()