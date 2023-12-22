import os
import time

import cv2
from ultralytics import YOLO


# vid_name = "traffic_vid_1.mp4"
vid_name = "both_side.mp4"


VIDEO_PATH_1 = f"./videos/{vid_name}"
VIDEO_PATH_2 = f"./videos/{vid_name}"
VIDEO_PATH_2 = f"./videos/{vid_name}"
VIDEO_PATH_4 = f"./videos/{vid_name}"



CAPTURE_INTERVAL = 5
CONFIDENCE_THRESHOLD = 0.4
YOLO_MODEL_LARGE = YOLO("yolo_models/yolov8l.pt")


OUTPUT_FOLDER = 'output'
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


try:
    frame_count = 0
    fps = int(round(video_1.get(cv2.CAP_PROP_FPS)))

    while video_1.isOpened() and video_2.isOpened() and video_3.isOpened() and video_4.isOpened():
        success_1, img_1 = video_1.read()
        success_2, img_2 = video_2.read()
        success_3, img_3 = video_3.read()
        success_4, img_4 = video_4.read()

        if not success_1 or not success_2 or not success_3 or not success_4:
            print("Can't receive frame (stream end?). Exiting ...")
            break


        if frame_count % (CAPTURE_INTERVAL * fps) == 0:
            cv2.imwrite(f"{OUTPUT_FOLDER}/camera-1.jpg", img_1)
            cv2.imwrite(f"{OUTPUT_FOLDER}/camera-2.jpg", img_2)
            cv2.imwrite(f"{OUTPUT_FOLDER}/camera-3.jpg", img_3)
            cv2.imwrite(f"{OUTPUT_FOLDER}/camera-4.jpg", img_4)

        frame_count += 1

        img_1 = cv2.resize(img_1, (640, 480))
        img_2 = cv2.resize(img_2, (640, 480))
        img_3 = cv2.resize(img_3, (640, 480))
        img_4 = cv2.resize(img_4, (640, 480))

        frame_1 = cv2.hconcat([img_1, img_2])
        frame_2 = cv2.hconcat([img_3, img_4])
        frame = cv2.vconcat([frame_1, frame_2])
        cv2.imshow('frame', frame)

        time.sleep(1/(fps*2))


        if cv2.waitKey(1) == ord('q'):
            break

finally:
    video_1.release()
    video_2.release()
    cv2.destroyAllWindows()
