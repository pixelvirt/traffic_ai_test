# pip install easydict

import numpy as np
import cv2
import torch
from ultralytics import YOLO


# Load video and video_info
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Unable to open video.")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

    return cap, frame_width, frame_height, fps


video_path = "videos/both_side.mp4"
video_path = "videos/traffic_vid_3.mp4"
cap, frame_width, frame_height, fps = get_video_info(video_path)

escape_frame, violation_frame = 2, 10


# Load model
def get_model(model_name):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = YOLO(model_name).to(device)
    return model


model_name = "yolo_models/yolov8n.pt"
model = get_model(model_name)


# Load coco labels
from coco_names import coco_class_list


from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


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
        use_cuda=False,
    )

    return deepsort


deepsort = initialize_deepsort()


# Line coordinates
line_coordinates = ((200, 0), (65, 845))


# Distance from line
def distance_from_line(x, y):
    (x1, y1), (x2, y2) = line_coordinates
    numerator = ((x2 - x1) * (y1 - y)) - ((x1 - x) * (y2 - y1))
    denominator = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    if denominator == 0:
        return 0
    
    return numerator / denominator


# Get direction of movement
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
def get_side_of_movement(point):
    x, y = point

    # Calculate distance from the line
    distance = distance_from_line(x, y)

    # Determine whether the object is on the left or right
    if distance >= 10:
        side = "R"
    elif distance <= -10:
        side = "L"
    else:
        side = "O"

    return side, distance


# Main code
try:
    frame_count = 0
    prev_detections = []
    violation_info = {}

    while True:
        success, frame = cap.read()
        if not success:
            print("Unable to read video frame.")
            exit()

        result = model.predict(frame, stream=True, conf=0.4, classes=[2, 3, 5, 7])
        result = list(result)[0]

        xywhs = result.boxes.xywh
        confs = result.boxes.conf
        classes = result.boxes.cls

        frame_count += 1

        if len(xywhs) == 0:
            continue

        outputs = deepsort.update(xywhs, confs, classes, frame)
        if len(outputs) > 0:
            bboxes_xyxy = outputs[:, :4]
            object_ids = outputs[:, -2]
            class_names = outputs[:, -1]
        
            for box, class_name, object_id in zip(bboxes_xyxy, class_names, object_ids):
                direction, side, distance = "", "", ""

                direction = get_direction_of_movement(prev_detections, box, object_id)

                cv2.line(frame, line_coordinates[0], line_coordinates[1], (0, 255, 0), 2)

                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                if direction == "F":
                    side, distance = get_side_of_movement((x1, np.mean([y1, y2])))
                elif direction == "B":
                    side, distance = get_side_of_movement((x2, np.mean([y1, y2])))

                if direction == "F" and side == "L":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(frame, f"{object_id} {direction} {side} {coco_class_list[class_name]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    v_r = violation_info.get(object_id)
                    if v_r:
                        violation_info[object_id].append(frame_count)
                    else:
                        violation_info[object_id] = [frame_count]

                elif direction == "B" and side == "R":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(frame, f"{object_id} {direction} {side} {coco_class_list[class_name]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    v_r = violation_info.get(object_id)
                    if v_r:
                        violation_info[object_id].append(frame_count)
                    else:
                        violation_info[object_id] = [frame_count]

                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.putText(frame, f"{object_id} {direction} {side} {coco_class_list[class_name]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                deleteable_keys = []
                for object_id, frames in violation_info.items():
                    if len(frames) < 2:
                        continue
                    elif frames[-1] - frames[-2] > escape_frame:
                        deleteable_keys.append(object_id)
                    elif len(frames) >= violation_frame:
                        deleteable_keys.append(object_id)
                        cv2.imwrite(f"output/violation-{frame_count}.jpg", frame)
                
                for key in deleteable_keys:
                    del violation_info[key]
                    

        prev_detections = outputs

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

except Exception as e:
    print(e)

finally:
    cap.release()
    cv2.destroyAllWindows()
