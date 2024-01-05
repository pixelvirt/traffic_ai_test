import os
import datetime
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from numba import jit

# Load coco labels
from coco_names import coco_class_list


# Line coordinates
line_coordinates = [((200, 1), (65, 845)), ((260,1), (1,625)), ((200, 1), (65, 845)), ((260,1), (1,625))]
escape_frame, violation_frame = 2, 5
violation_allowed_limit = 10


# Cuda availability
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
is_cuda_available = True if device == torch.device("cuda") else False


# Initialize deep sort
def initialize_deepsort():
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(
        cfg_deep.DEEPSORT.REID_CKPT,
        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg_deep.DEEPSORT.MAX_AGE,
        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
        use_cuda=is_cuda_available,
    )

    return deepsort

deepsort_1 = initialize_deepsort()
deepsort_2 = initialize_deepsort()
deepsort_3 = initialize_deepsort()
deepsort_4 = initialize_deepsort()
deepsort = [deepsort_1, deepsort_2, deepsort_3, deepsort_4]


# Load model
def get_model(model_name, device):
    model = YOLO(model_name).to(device)
    return model

model_name = "yolo_models/yolov8l.pt"
model = get_model(model_name, device)


# Load video and video_info
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Unable to open video.")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

    return cap, frame_width, frame_height, fps


# Distance from line
@jit(target_backend="cuda" if is_cuda_available else "cpu")
def distance_from_line(x, y, line_coordinate):
    (x1, y1), (x2, y2) = line_coordinate
    numerator = ((x2 - x1) * (y1 - y)) - ((x1 - x) * (y2 - y1))
    denominator = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    if denominator == 0:
        return 0
    
    return numerator / denominator


# Get direction of movement
@jit(target_backend="cuda")
def get_direction_of_movement(prev_detections, box, object_id):
    direction = ""

    prev = [item for item in prev_detections if item[-2] == object_id]
    if prev:
        prev = prev[0]
        if prev[1] < box[1] and prev[3] < box[3]:
            direction = "F"
        elif prev[1] > box[1] and prev[3] > box[3]:
            direction = "B"
        else:
            direction = "S"

    return direction


# Get side of movement
@jit(target_backend="cuda")
def get_side_of_movement(point, line_coordinate):
    x, y = point

    # Calculate distance from the line
    distance = distance_from_line(x, y, line_coordinate)

    # Determine whether the object is on the left or right
    if distance >= 10:
        side = "R"
    elif distance <= -10:
        side = "L"
    else:
        side = "O"

    return side, distance


video_path_1 = "videos/traffic_vid_3.mp4"
video_path_2 = "videos/traffic_vid_2.mp4"
cap_1, frame_width_1, frame_height_1, fps_1 = get_video_info(video_path_1)
cap_2, frame_width_2, frame_height_2, fps_2 = get_video_info(video_path_2)
cap_3, frame_width_3, frame_height_3, fps_3 = get_video_info(video_path_1)
cap_4, frame_width_4, frame_height_4, fps_4 = get_video_info(video_path_2)

try:
    frame_count = 0
    prev_detections = []
    temp_prev_detections = []
    violated_object_ids = []
    violation_info = {}

    while True:
        success_1, frame_1 = cap_1.read()
        success_2, frame_2 = cap_2.read()
        success_3, frame_3 = cap_3.read()
        success_4, frame_4 = cap_4.read()

        

        if not success_1 or not success_2:
            raise ValueError("Error: Unable to read video frame.")
        
        frame_count += 1
        if frame_count % escape_frame != 0:
            continue

        # Get detections
        frames_list = [frame_1, frame_2, frame_3, frame_4]
        for idx, frame in enumerate(frames_list):
            if not os.path.exists(f"output/{idx}"):
                os.mkdir(f"output/{idx}")

        detections = model.predict(frames_list, stream=True, conf=0.4, classes=[2, 3, 5, 7])
        detections = list(detections)

        for idx, detection in enumerate(detections):
            xywhs = detection.boxes.xywh.cpu().numpy()
            confs = detection.boxes.conf.cpu().numpy()
            classes = detection.boxes.cls.cpu().numpy()

            cv2.line(frames_list[idx], line_coordinates[idx][0], line_coordinates[idx][1], (0, 255, 0), 2)

            if len(xywhs) == 0:
                continue

            # Pass detections to deepsort
            trackings = deepsort[idx].update(xywhs, confs, classes, frames_list[idx])
            if len(trackings) > 0:
                bboxes_xyxy = trackings[:, :4]
                object_ids = trackings[:, -2]
                class_names = trackings[:, -1]

                for box, class_name, object_id in zip(bboxes_xyxy, class_names, object_ids):
                    direction, side, distance = "", "", 0

                    try:
                        direction = get_direction_of_movement(prev_detections[idx], box, object_id)
                    except:
                        direction = get_direction_of_movement([], box, object_id)

                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    if direction == "F":
                        side, distance = get_side_of_movement((x1, np.mean([y1, y2])), line_coordinates[idx])
                    elif direction == "B":
                        side, distance = get_side_of_movement((x2, np.mean([y1, y2])), line_coordinates[idx])
                    elif direction == "S":
                        side, distance = get_side_of_movement((np.mean([x1, x2]), np.mean([y1, y2])), line_coordinates[idx])

                    if direction == "F" and side == "L" and (distance > violation_allowed_limit):
                        cv2.rectangle(frames_list[idx], (x1, y1), (x2, y2), (0, 0, 255), 1)
                        cv2.putText(frames_list[idx], f"{object_id} {direction} {side} {coco_class_list[class_name]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                        v_r = violation_info.get(object_id)
                        if v_r:
                            violation_info[object_id].append(frame_count)
                        else:
                            violation_info[object_id] = [frame_count]

                    elif direction == "B" and side == "R" and (distance > violation_allowed_limit):
                        cv2.rectangle(frames_list[idx], (x1, y1), (x2, y2), (0, 0, 255), 1)
                        cv2.putText(frames_list[idx], f"{object_id} {direction} {side} {coco_class_list[class_name]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                        v_r = violation_info.get(object_id)
                        if v_r:
                            violation_info[object_id].append(frame_count)
                        else:
                            violation_info[object_id] = [frame_count]

                    else:
                        cv2.rectangle(frames_list[idx], (x1, y1), (x2, y2), (255, 0, 0), 1)
                        cv2.putText(frames_list[idx], f"{object_id} {direction} {side} {coco_class_list[class_name]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    deleteable_keys = []
                    for object_id, frames in violation_info.items():
                        if len(frames) < 2:
                            continue
                        elif frames[-1] - frames[-2] > escape_frame:
                            deleteable_keys.append(object_id)
                        elif len(frames) >= violation_frame:
                            if object_id not in violated_object_ids:
                                violated_object_ids.append(object_id)
                                with open("violation_data.txt", "a") as f:
                                    f.write(f"camera-{idx} {object_id} {datetime.datetime.now().time()}\n")
                                deleteable_keys.append(object_id)
                                cv2.imwrite(f"output/{idx}/violation-{frame_count}.jpg", frames_list[idx])
                    
                    for key in deleteable_keys:
                        del violation_info[key]

            temp_prev_detections.append(trackings)

        prev_detections = temp_prev_detections
        temp_prev_detections = []

        
        if frame_width_1 != frame_width_2 or frame_height_1 != frame_height_2:
            frame_2 = cv2.resize(frame_2, (frame_width_1, frame_height_1))

        if frame_width_1 != frame_width_4 or frame_height_1 != frame_height_4:
            frame_2 = cv2.resize(frame_4, (frame_width_1, frame_height_1))

        display_frames = cv2.hconcat([item for item in frames_list])
        cv2.imshow("Output", display_frames)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break    

finally:
    cap_1.release()
    cap_2.release()
    cap_3.release()
    cap_4.release()
    cv2.destroyAllWindows()