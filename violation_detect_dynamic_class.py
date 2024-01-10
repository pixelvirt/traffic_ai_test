import os
import datetime
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from numba import jit

from coco_names import coco_class_list


def jit_cuda_decorator(func):
    def wrapper(self, *args, **kwargs):
        use_cuda = getattr(self, 'use_cuda', False)
        if use_cuda:
            return jit(target_backend="cuda")(func)(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)
    return wrapper


class Traffic:
    def __init__(
            self, 
            video_paths, 
            model_name, 
            mid_line_coordinates, 
            red_line_coordinates, 
            escape_frame = 2, 
            violation_frame = 5,
            violation_allowed_limit = 10,
        ):
        self.video_paths = video_paths
        self.videos = [self.get_video_info(video_path) for video_path in video_paths]
        for idx, cap in enumerate(self.videos):
            if not cap[0].isOpened():
                raise ValueError(f"Error: Unable to open video {idx}.")

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.is_cuda_available = True if self.device == torch.device("cuda") else False
        self.use_cuda = True if self.device == torch.device("cuda") else False

        self.deepsort = [self.initialize_deepsort() for _ in range(len(video_paths))]

        self.model_name = model_name
        self.model = self.get_model()

        self.coco_class_list = coco_class_list

        self.mid_line_coordinates = mid_line_coordinates
        self.red_line_coordinates = red_line_coordinates
        if not len(mid_line_coordinates) == len(self.videos) or not len(red_line_coordinates) == len(self.videos):
            raise ValueError("Error: Line coordinates and video count must be equal.")

        self.red_light_on = False
        self.escape_frame = escape_frame 
        self.violation_frame = violation_frame
        self.violation_allowed_limit = violation_allowed_limit

        self.frame_count = 0
        self.violation_info = {}
        self.violated_object_ids = []
        self.red_light_violation_ids = []

        self.prev_detections = []
        self.temp_prev_detections = []

        self.create_output_folders()

    def create_output_folders(self):
        if not os.path.exists("output"):
            os.mkdir("output")
        if not os.path.exists("data"):
            os.mkdir("data")

    def get_video_info(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Unable to open video.")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

        return cap, frame_width, frame_height, fps
    
    
    def initialize_deepsort(self):
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
            use_cuda=self.is_cuda_available,
        )

        return deepsort
    
    
    def get_model(self):
        model = YOLO(self.model_name).to(self.device)
        return model
    
    @jit(target_backend="cuda")
    def get_direction_of_movement(self, prev_detections, box, object_id):
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
    

    @jit(target_backend="cuda")
    def distance_from_line(self, x, y, line_coordinate):
        (x1, y1), (x2, y2) = line_coordinate
        numerator = ((x2 - x1) * (y1 - y)) - ((x1 - x) * (y2 - y1))
        denominator = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    @jit(target_backend="cuda")
    def get_side_of_movement(self, point, line_coordinate):
        x, y = point

        # Calculate distance from the line
        distance = self.distance_from_line(x, y, line_coordinate)

        # Determine whether the object is on the left or right
        if distance >= 10:
            side = "R"
        elif distance <= -10:
            side = "L"
        else:
            side = "O"

        return side, distance
    

    def red_light_violation(self, idx, frames_list, x1, y1, x2, y2, object_id, class_name, direction):
        if direction == "F" and x2 >= self.red_line_coordinates[idx][0] and y2 >= self.red_line_coordinates[idx][1]:
            cv2.rectangle(frames_list[idx], (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frames_list[idx], f"{object_id} {direction} {coco_class_list[class_name]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if object_id not in self.red_light_violation_ids:
                self.red_light_violation_ids.append(object_id)
                with open("data/red_light_data.txt", "a") as f:
                    f.write(f"camera-{idx} {object_id} {datetime.datetime.now().time()}\n")
                cv2.imwrite(f"output/{idx}/red_light_violation-{self.frame_count}.jpg", frames_list[idx])
    

    def lane_violation_detection(self, idx, detection, frames_list, frame_width):
        xywhs = detection.boxes.xywh.cpu().numpy()
        confs = detection.boxes.conf.cpu().numpy()
        classes = detection.boxes.cls.cpu().numpy()

        cv2.line(frames_list[idx], self.mid_line_coordinates[idx][0], self.mid_line_coordinates[idx][1], (0, 255, 0), 2)
        if self.red_light_on:
            cv2.line(frames_list[idx], self.red_line_coordinates[idx], (frame_width, self.red_line_coordinates[idx][1]), (0, 0, 255), 2)

        if len(xywhs) == 0:
            return None

        # Pass detections to deepsort
        trackings = self.deepsort[idx].update(xywhs, confs, classes, frames_list[idx])
        if len(trackings) > 0:
            bboxes_xyxy = trackings[:, :4]
            object_ids = trackings[:, -2]
            class_names = trackings[:, -1]

            for box, class_name, object_id in zip(bboxes_xyxy, class_names, object_ids):
                direction, side, distance = "", "", 0

                try:
                    direction = self.get_direction_of_movement(self.prev_detections[idx], box, object_id)
                except:
                    direction = self.get_direction_of_movement([], box, object_id)

                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                if direction == "F":
                    side, distance = self.get_side_of_movement((x1, np.mean([y1, y2])), self.mid_line_coordinates[idx])
                elif direction == "B":
                    side, distance = self.get_side_of_movement((x2, np.mean([y1, y2])), self.mid_line_coordinates[idx])
                elif direction == "S":
                    side, distance = self.get_side_of_movement((np.mean([x1, x2]), np.mean([y1, y2])), self.mid_line_coordinates[idx])

                if direction == "F" and side == "L" and (distance > self.violation_allowed_limit):
                    cv2.rectangle(frames_list[idx], (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(frames_list[idx], f"{object_id} {direction} {side} {coco_class_list[class_name]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    v_r = self.violation_info.get(object_id)
                    if v_r:
                        self.violation_info[object_id].append(self.frame_count)
                    else:
                        self.violation_info[object_id] = [self.frame_count]

                elif direction == "B" and side == "R" and (distance > self.violation_allowed_limit):
                    cv2.rectangle(frames_list[idx], (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(frames_list[idx], f"{object_id} {direction} {side} {coco_class_list[class_name]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    v_r = self.violation_info.get(object_id)
                    if v_r:
                        self.violation_info[object_id].append(self.frame_count)
                    else:
                        self.violation_info[object_id] = [self.frame_count]

                else:
                    cv2.rectangle(frames_list[idx], (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.putText(frames_list[idx], f"{object_id} {direction} {side} {coco_class_list[class_name]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                if self.red_light_on:
                    self.red_light_violation(idx, frames_list, x1, y1, x2, y2, object_id, class_name, direction)

                deleteable_keys = []
                for object_id, frames in self.violation_info.items():
                    if len(frames) < 2:
                        continue
                    elif frames[-1] - frames[-2] > self.escape_frame:
                        deleteable_keys.append(object_id)
                    elif len(frames) >= self.violation_frame:
                        if object_id not in self.violated_object_ids:
                            self.violated_object_ids.append(object_id)
                            with open("data/violation_data.txt", "a") as f:
                                f.write(f"camera-{idx} {object_id} {datetime.datetime.now().time()}\n")
                            deleteable_keys.append(object_id)
                            cv2.imwrite(f"output/{idx}/violation-{self.frame_count}.jpg", frames_list[idx])
                
                for key in deleteable_keys:
                    del self.violation_info[key]

        return trackings
    

    def run(self):
        try:
            while True:
                frames_list = []
                for cap, frame_width, frame_height, fps in self.videos:
                    success, frame = cap.read()
                    frames_list.append(frame)

                self.frame_count += 1
                if self.frame_count % self.escape_frame != 0:
                    continue

                # temporary --> change this
                if 200 < self.frame_count < 400:
                    self.red_light_on = True
                else:
                    self.red_light_on = False

                # Get detections
                for idx, frame in enumerate(frames_list):
                    if not os.path.exists(f"output/{idx}"):
                        os.mkdir(f"output/{idx}")

                detections = self.model.predict(frames_list, stream=True, conf=0.4, classes=[2, 3, 5, 7])
                detections = list(detections)

                for idx, detection in enumerate(detections):
                    trackings = self.lane_violation_detection(idx, detection, frames_list, frame_width)
                    if trackings is not None:
                        self.temp_prev_detections.append(trackings)

                self.prev_detections = self.temp_prev_detections
                self.temp_prev_detections = []

                
                # Display output
                for idx, frame in enumerate(frames_list):
                    if idx == 0:
                        continue
                    elif self.videos[idx][1] != self.videos[0][1] or self.videos[idx][2] != self.videos[0][2]:
                        cv2.resize(frame, (self.videos[0][1], self.videos[0][2]))

                display_frames = cv2.hconcat([item for item in frames_list])
                cv2.imshow("Output", display_frames)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break 

        finally:
            pass

    def __del__(self):
        for cap in self.videos:
            cap[0].release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_paths = ["videos/traffic_vid_3.mp4", "videos/traffic_vid_2.mp4", "videos/traffic_vid_3.mp4"]
    model_name = "yolo_models/yolov8n.pt"
    mid_line_coordinates = [((200, 1), (65, 845)), ((260,1), (1,625)), ((200, 1), (65, 845))]
    red_line_coordinates = [(80,790), (1, 650), (80,790)]

    traffic_obj = Traffic(video_paths, model_name, mid_line_coordinates, red_line_coordinates)
    traffic_obj.run()
