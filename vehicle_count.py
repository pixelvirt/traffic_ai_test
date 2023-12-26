import math
from collections import Counter

import cv2
from ultralytics import YOLO

from coco_names import coco_class_list, vehicle_class_list


model = YOLO("yolo_models/yolov8l.pt")
show_frames = True


# vid_name = "traffic_vid_1.mp4"
vid_name = "both_side.mp4"


video_1 = cv2.VideoCapture(f"./videos/{vid_name}")
video_2 = cv2.VideoCapture(f"./videos/{vid_name}")
video_3 = cv2.VideoCapture(f"./videos/{vid_name}")
video_4 = cv2.VideoCapture(f"./videos/{vid_name}")


if not video_1.isOpened() or not video_2.isOpened() or not video_3.isOpened() or not video_4.isOpened():
    print("Cannot open video")
    exit()


try:
    frame_count = 0
    fps = int(round(video_1.get(cv2.CAP_PROP_FPS)))
    time_interval = 5

    while video_1.isOpened() and video_2.isOpened() and video_3.isOpened() and video_4.isOpened():
        success_1, img_1 = video_1.read()
        success_2, img_2 = video_2.read()
        success_3, img_3 = video_3.read()
        success_4, img_4 = video_4.read()

        if not success_1 or not success_2 or not success_3 or not success_4:
            break

        if show_frames:
            frame_1 = cv2.hconcat([img_1, img_2])
            frame_2 = cv2.hconcat([img_3, img_4])
            frame = cv2.vconcat([frame_1, frame_2])
            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_count % (fps * time_interval) == 0:
            results = model([img_1, img_2, img_3, img_4], conf=0.5, classes=[2, 3, 5, 7])
            for idx, r in enumerate(results):
                cls = r.boxes.cls
                object_classes = [coco_class_list[int(item)] for item in cls]

                counter = dict(Counter(object_classes))
                vehicle_count = 0
                for key in counter:
                    if key in vehicle_class_list:
                        vehicle_count += counter[key]
                
                counter["vehicle"] = vehicle_count

                with open(f"data.txt", 'a') as f:
                    f.write(f"video-{idx}: {counter}\n")

        frame_count += 1

finally:
    video_1.release()
    video_2.release()
    video_3.release()
    video_4.release()
    cv2.destroyAllWindows()