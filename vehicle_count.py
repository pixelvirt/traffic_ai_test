import math
from collections import Counter

import cv2
from ultralytics import YOLO

from coco_names import coco_class_list, vehicle_class_list


model = YOLO("yolo_models/yolov8l.pt")
show_frames = True

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Unable to open video.")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

    return cap, frame_width, frame_height, fps


# vid_name = "traffic_vid_1.mp4"
vid_name = "both_side.mp4"

video_paths = [f"videos/{vid_name}" for _ in range(4)]
cap_list = [get_video_info(video_path) for video_path in video_paths]


try:
    frame_count = 0
    fps = int(round(cap_list[0][3]))
    time_interval = 5

    while True:
        frames_list = []
        for cap, frame_width, frame_height, fps in cap_list:
            success, frame = cap.read()
            if not success:
                print("Error: Unable to read frame.")
                break
            frames_list.append(frame)

        if show_frames:
            display_frames = cv2.hconcat([item for item in frames_list])
            cv2.imshow("Output", display_frames)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_count % (fps * time_interval) == 0:
            results = model(frames_list, conf=0.5, classes=[2, 3, 5, 7])
            for idx, r in enumerate(results):
                cls = r.boxes.cls
                object_classes = [coco_class_list[int(item)] for item in cls]

                counter = dict(Counter(object_classes))
                vehicle_count = 0
                for key in counter:
                    if key in vehicle_class_list:
                        vehicle_count += counter[key]
                
                counter["vehicle"] = vehicle_count

                with open(f"data/data.txt", 'a') as f:
                    f.write(f"video-{idx}: {counter}\n")

        frame_count += 1

finally:
    for cap in cap_list:
        cap[0].release()
    cv2.destroyAllWindows()