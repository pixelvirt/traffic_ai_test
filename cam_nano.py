# import required modules
import os
import math
from collections import Counter

import cv2
from ultralytics import YOLO

from coco_names import coco_class_list, vehicle_class_list


# define constants
OUTPUT_FOLDER = 'output'
CAPTURE_INTERVAL = 5
CONFIDENCE_THRESHOLD = 0.75
YOLO_MODEL_NANO = YOLO("yolo_models/yolov8n.pt")
YOLO_MODEL_LARGE = YOLO("yolo_models/yolov8l.pt")


# create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


# open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


# main loop
try:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    frame_count = 0

    while True:
        # read frame from webcam
        success, img = cap.read()
        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # run YOLO model on each frame
        frames = YOLO_MODEL_NANO(img, stream=True)

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
                    cv2.putText(img, f"{class_name}: {count}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    text_y += 15

                cv2.imwrite(f"{OUTPUT_FOLDER}/{frame_count}.jpg", img)
                print(f"Captured {frame_count}.jpg")
                
            detected_classes = []

        cv2.imshow("Webcam", img)

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
                

finally:
    cap.release()
    cv2.destroyAllWindows()
