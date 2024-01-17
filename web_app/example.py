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


# Load model
def get_model(model_name, device):
    model = YOLO(model_name).to(device)
    return model


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
def distance_from_line(x, y, line_coordinate):
    (x1, y1), (x2, y2) = line_coordinate
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
def get_side_of_movement(point, line_coordinate, violation_allowed_limit_px=10):
    x, y = point

    # Calculate distance from the line
    distance = distance_from_line(x, y, line_coordinate)

    # Determine whether the object is on the left or right
    if distance >= violation_allowed_limit_px:
        side = "R"
    elif distance <= -violation_allowed_limit_px:
        side = "L"
    else:
        side = "O"

    return side, distance


# Red light violationclear
def red_light_violation(red_line_coordinates, frame_count, frames_list, idx, x1, y1, x2, y2, object_id, direction, red_light_violation_ids, violation_allowed_limit_px=10):
    if direction == "F" and x2 >= red_line_coordinates[idx][0] + violation_allowed_limit_px and y2 >= red_line_coordinates[idx][1] + violation_allowed_limit_px:
        cv2.rectangle(frames_list[idx], (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(frames_list[idx], f"{object_id} red light violation", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if object_id not in red_light_violation_ids:
            red_light_violation_ids.append(object_id)
            with open("data/red_light_data.txt", "a") as f:
                f.write(f"camera-{idx} {object_id} {datetime.datetime.now().time()}\n")
            cv2.imwrite(f"output/{idx}/red_light_violation-{object_id}-{frame_count}.jpg", frames_list[idx])


# Parking violation
def parking_violation(detections, frame_count, idx, frames_list, prev_positions_list, violated_object_ids, num_prev_positions=5):
    global coco_class_list

    if not len(prev_positions_list) == num_prev_positions:
        return

    object_ids = []
    has_modved = []
    prev_detections_object_ids = []

    for detection in detections:
        object_id = detection[-2]
        object_ids.append(object_id)
        position = detection[:4]

        for prev_positions in prev_positions_list[:-1]:
            for prev_position in prev_positions:
                for item in prev_position:
                    if item[-2] == object_id:
                        if item[-2] not in prev_detections_object_ids:
                            prev_detections_object_ids.append(item[-2])
                        prev_pos = item[:4]
                        if not np.array_equal(prev_pos, position):
                            has_modved.append(object_id)
                            break
    
    extra_object_ids = list(set(object_ids) - set(prev_detections_object_ids))
    not_moved = list((set(object_ids) - set(has_modved)) - set(extra_object_ids))

    for object_id in not_moved:
        for item in detections:
            if item[-2] == object_id:
                class_name = item[-1]
                x1, y1, x2, y2 = item[:4]
                cv2.rectangle(frames_list[idx], (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(frames_list[idx], f"{object_id} parking violation", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if object_id not in violated_object_ids:
            violated_object_ids.append(object_id)
            with open("data/parking_violation_data.txt", "a") as f:
                f.write(f"camera-{idx} {object_id} {datetime.datetime.now().time()}\n")
            cv2.imwrite(f"output/{idx}/parking_violation-{object_id}-{frame_count}.jpg", frames_list[idx])


# Create output folders
def create_output_folders():
    if not os.path.exists("output"):
        os.mkdir("output")
    if not os.path.exists("data"):
        os.mkdir("data")


def main(
        cap_list, 
        model, 
        deepsort, 
        mid_line_coordinates, 
        red_line_coordinates, 
        violation_allowed_limit_px = 10, 
        escape_frame = 2, 
        violation_frame = 5, 
        check_red_light_violation = False, 
        check_parking_violation = False, 
        num_prev_positions = 5,
    ):
    try:
        frame_count = 0
        prev_detections = []
        temp_prev_detections = []
        violated_object_ids = []
        violation_info = {}
        red_light_violation_ids = []
        prev_positions_list = []

        while True:
            frames_list = []
            for cap, frame_width, frame_height, fps in cap_list:
                success, frame = cap.read()
                frames_list.append(frame)

            frame_count += 1
            if frame_count % escape_frame != 0:
                continue

            # change this code to change the red light logic
            if 200 < frame_count < 400:
                red_light = True
            else:
                red_light = False

            for idx, frame in enumerate(frames_list):
                if not os.path.exists(f"output/{idx}"):
                    os.mkdir(f"output/{idx}")

            detections = model(frames_list, stream=True, conf=0.4, classes=[2, 3, 5, 7])
            detections = list(detections)

            for idx, detection in enumerate(detections):
                xywhs = detection.boxes.xywh.cpu().numpy()
                confs = detection.boxes.conf.cpu().numpy()
                classes = detection.boxes.cls.cpu().numpy()

                cv2.line(frames_list[idx], mid_line_coordinates[idx][0], mid_line_coordinates[idx][1], (0, 255, 0), 2)
                if red_light:
                    cv2.line(frames_list[idx], red_line_coordinates[idx], (frame_width, red_line_coordinates[idx][1]), (0, 0, 255), 2)

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
                            side, distance = get_side_of_movement((x1, np.mean([y1, y2])), mid_line_coordinates[idx], violation_allowed_limit_px)
                        elif direction == "B":
                            side, distance = get_side_of_movement((x2, np.mean([y1, y2])), mid_line_coordinates[idx], violation_allowed_limit_px)
                        elif direction == "S":
                            side, distance = get_side_of_movement((np.mean([x1, x2]), np.mean([y1, y2])), mid_line_coordinates[idx], violation_allowed_limit_px)

                        if direction == "F" and side == "L":
                            cv2.rectangle(frames_list[idx], (x1, y1), (x2, y2), (0, 0, 255), 1)
                            cv2.putText(frames_list[idx], f"{object_id} {distance:.1f} lane violation", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                            v_r = violation_info.get(object_id)
                            if v_r:
                                violation_info[object_id].append(frame_count)
                            else:
                                violation_info[object_id] = [frame_count]

                        elif direction == "B" and side == "R":
                            cv2.rectangle(frames_list[idx], (x1, y1), (x2, y2), (0, 0, 255), 1)
                            cv2.putText(frames_list[idx], f"{object_id} {distance:.1f} lane violation", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                            v_r = violation_info.get(object_id)
                            if v_r:
                                violation_info[object_id].append(frame_count)
                            else:
                                violation_info[object_id] = [frame_count]

                        else:
                            cv2.rectangle(frames_list[idx], (x1, y1), (x2, y2), (255, 0, 0), 1)
                            # cv2.putText(frames_list[idx], f"{object_id} {direction} {side} {coco_class_list[class_name]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                        if check_red_light_violation and red_light:
                            red_light_violation(red_line_coordinates, frame_count, frames_list, idx, x1, y1, x2, y2, object_id, direction, red_light_violation_ids)

                        if check_parking_violation:
                            parking_violation(trackings, frame_count, idx, frames_list, prev_positions_list, violated_object_ids, num_prev_positions)

                        deleteable_keys = []
                        for object_id, frames in violation_info.items():
                            if len(frames) < 2:
                                continue
                            elif frames[-1] - frames[-2] > escape_frame:
                                deleteable_keys.append(object_id)
                            elif len(frames) >= violation_frame:
                                if object_id not in violated_object_ids:
                                    violated_object_ids.append(object_id)
                                    with open("data/lane_violation_data.txt", "a") as f:
                                        f.write(f"camera-{idx} {object_id} {datetime.datetime.now().time()}\n")
                                    deleteable_keys.append(object_id)
                                    cv2.imwrite(f"output/{idx}/lane-violation-{object_id}-{frame_count}.jpg", frames_list[idx])
                        
                        for key in deleteable_keys:
                            del violation_info[key]

                temp_prev_detections.append(trackings)

            prev_detections = temp_prev_detections
            temp_prev_detections = []

            # Save previous positions of objects every second for parking violation detection
            if check_parking_violation and frame_count % fps == 0:
                prev_positions_list.append(prev_detections)
                if len(prev_positions_list) > num_prev_positions:
                    prev_positions_list.pop(0)
            
            # Display output
            for idx, frame in enumerate(frames_list):
                if idx == 0:
                    continue
                elif len(cap_list) > 1 and (cap_list[idx][1] != cap_list[0][1] or cap_list[idx][2] != cap_list[0][2]):
                    cv2.resize(frame, (cap_list[0][1], cap_list[0][2]))

            display_frames = cv2.hconcat([item for item in frames_list])
            # cv2.imshow("Output", display_frames)

            # yield to display to web
            ret, jpeg = cv2.imencode('.jpg', display_frames)
            display_frames = jpeg.tobytes()
            yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + display_frames + b'\r\n')
            # yield ([b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'] for frame in display_frames)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break    

    finally:
        for cap in cap_list:
            cap[0].release()
        cv2.destroyAllWindows()


# # Main Logic
# model_name = "yolo_models/yolov8n.pt"
# model = get_model(model_name, device)

# # video_paths = ["videos/traffic_vid_3.mp4", "videos/traffic_vid_2.mp4", "videos/traffic_vid_3.mp4"]
# video_paths = ["videos/traffic_vid_3.mp4", "videos/traffic_vid_2.mp4"]
# deepsort = [initialize_deepsort() for _ in range(len(video_paths))]
# cap_list = [get_video_info(video_path) for video_path in video_paths]

# for idx, cap in enumerate(cap_list):
#     if not cap[0].isOpened():
#         raise ValueError(f"Error: Unable to open video {idx}.")

# # mid_line_coordinates = [((200, 1), (65, 845)), ((260,1), (1,625)), ((200, 1), (65, 845))]
# # red_line_coordinates = [(80,790), (1, 650), (80,790)]
# mid_line_coordinates = [((200, 1), (65, 845)), ((260,1), (1,625))]
# red_line_coordinates = [(80,790), (1, 650)]
# check_parking_violation = True
# check_red_light_violation = True
# red_light = False
# escape_frame = 2
# violation_frame = 5
# violation_allowed_limit_px = 10    # pixel from the line up to which the violation is allowed
# num_prev_positions = 5              # number of previous positions to consider for detection of parking violation

# if not len(mid_line_coordinates) == len(cap_list) or not len(red_line_coordinates) == len(cap_list):
#     raise ValueError("Error: Line coordinates and video count must be equal.")

# create_output_folders()
# main(cap_list, model, deepsort, mid_line_coordinates, red_line_coordinates, violation_allowed_limit_px, escape_frame, violation_frame, check_red_light_violation, check_parking_violation, num_prev_positions)
