import os
import datetime

import cv2
import torch
import numpy as np
from ultralytics import YOLO

from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config

from coco_names import coco_class_list



class TrafficViolationDetection:
    def __init__(
        self,                                   # represents the current instance of the class
        video_paths,                            # list of paths for traffic violation detection (may be list of location of files or camera urls)
        model_name,                             # name of the yolo model to be used for detection
        lane_violation_line_coordinates,        # coordinates of the line for lane violation detection
        red_light_violation_line_coordinates,   # coordinates of the line for red light violation detection
        escape_frame = 2,                       # number of frames to skip
        allowed_violation_frame_count = 5,      # number of frames allowed for violation
        violation_allowed_limit_px = 10,        # number of pixels allowed for violation in case of lane violation and red light violation
    ):
        # Initialize videos from video paths and check if the provided video paths are working or not
        self.video_paths = video_paths,
        self.videos = [self.get_video_info(path) for path in video_paths]
            
        # Check if cuda is available or not
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.is_cuda_available = True if self.device == torch.device("cuda") else False

        # Load the model
        self.model = YOLO(model_name).to(self.device)
        self.model.fuse()

        # Initialize the confidence and track classes
        # edit this if necessary
        self.conf = 0.4
        self.track_classes = [1, 2, 3, 5, 7]

        # Initialize the deep sort model for every video
        self.deepsorts = [self.initialize_deepsort() for _ in range(len(self.videos))]

        # Load the coco class list
        self.classes_list = coco_class_list

        # Initialize the lane violation line coordinates
        self.lane_violation_line_coordinates = lane_violation_line_coordinates
        self.red_light_violation_line_coordinates = red_light_violation_line_coordinates
        if not len(self.lane_violation_line_coordinates) == len(self.red_light_violation_line_coordinates) == len(self.videos):
            raise ValueError("Error: lane_violation_line_coordinates and  red_light_violation_line_coordinates must be provided for every video.")
        
        # Initialize red light for every video
        self.red_lights = [False for _ in range(len(self.videos))]

        # Initialize the type of violation to detect
        self.detect_lane_violation = True
        self.detect_red_light_violation = True
        self.detect_parking_violation = True
        
        # Initialize the escape frame and allowed violation frame
        self.escape_frame = escape_frame
        self.allowed_violation_frame_count = allowed_violation_frame_count
        self.violation_allowed_limit_px = violation_allowed_limit_px

        # Initialize the prev frames and prev detections
        self.save_length = 60       # number of frames and detections to save (edit this if necessary)
        self.prev_frames = []
        self.prev_detections = []
        self.temp_prev_detections = []

        # Initialize violation data
        self.lane_violation_object_ids = [[] for _ in range(len(self.videos))]
        self.lane_violation_object_info = {i: {} for i in range(len(self.videos))}
        self.red_light_violation_object_ids = [[] for _ in range(len(self.videos))]
        self.parking_violation_object_ids = [[] for _ in range(len(self.videos))]
        self.parking_violation_object_info = {i: {} for i in range(len(self.videos))}

        # For violation video writing
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")

        # Initialize video chunk writers for every video and violations
        self.video_chunk_writers = None
        self.video_chunk_size = 5
        self.violation_video_chunk_writers = {i: {} for i in range(len(self.videos))}

        # Some extra variables
        self.frame_count = 0
        self.sample_fps = 30
        self.draw_lines = True
        self.show_video_output = True
        self.window_size = (720, 1280)

        self.create_output_folders()


    # Get video information (capture is opened/valid or not, frames per second, video width, video height)
    def get_video_info(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Error: Unable to open video at {path}.")
        
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return cap, fps, width, height
    
    
    # Initialize the deep sort model for object tracking
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
    

    # Create output folders if does not exist other wise writing to output folder will not work
    def create_output_folders(self):
        if not os.path.exists("output"):
            os.mkdir("output")
        if not os.path.exists("output/data"):
            os.mkdir("output/data")
        if not os.path.exists("output/lane_violation"):
            os.mkdir("output/lane_violation")
        if not os.path.exists("output/red_light_violation"):
            os.mkdir("output/red_light_violation")
        if not os.path.exists("output/parking_violation"):
            os.mkdir("output/parking_violation")
        if not os.path.exists("output/video_chunks"):
            os.mkdir("output/video_chunks")
        if not os.path.exists("output/violation_video_chunks"):
            os.mkdir("output/violation_video_chunks")

        # Create output folders for every video camera
        for idx, _ in enumerate(self.videos):
            if not os.path.exists(f"output/data/{idx}"):
                os.mkdir(f"output/data/{idx}")
            if not os.path.exists(f"output/lane_violation/{idx}"):
                os.mkdir(f"output/lane_violation/{idx}")
            if not os.path.exists(f"output/red_light_violation/{idx}"):
                os.mkdir(f"output/red_light_violation/{idx}")
            if not os.path.exists(f"output/parking_violation/{idx}"):
                os.mkdir(f"output/parking_violation/{idx}")
            if not os.path.exists(f"output/video_chunks/{idx}"):
                os.mkdir(f"output/video_chunks/{idx}")
            if not os.path.exists(f"output/violation_video_chunks/{idx}"):
                os.mkdir(f"output/violation_video_chunks/{idx}")


    # Calculate the distance of a point from a line 
    def distance_from_line(self, x, y, line_coordinate):
        (x1, y1), (x2, y2) = line_coordinate
        numerator = ((x2 - x1) * (y1 - y)) - ((x1 - x) * (y2 - y1))
        denominator = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if denominator == 0:
            return 0
        
        return numerator / denominator
    

    # Get the side of movement of an object (left, right or none)
    def get_side_of_movement(self, point, line_coordinate):
        x, y = point

        # Calculate distance from the line
        distance = self.distance_from_line(x, y, line_coordinate)

        # Determine whether the object is on the left or right
        if distance >= self.violation_allowed_limit_px:
            side = "R"
        elif distance <= -(self.violation_allowed_limit_px):
            side = "L"
        else:
            side = "N"

        return side, distance
    

    # Get the direction of movement of an object (forward, backward or none)
    def get_direction_of_movement(self, box, obj_id, frame_idx):
        direction = "N"

        if len(self.prev_detections) > 3:
            prev_detection_all_list = self.prev_detections[-1]
            if len(prev_detection_all_list) > frame_idx:
                prev_detection_frame_list = prev_detection_all_list[frame_idx]
                prev_detection = next((item for item in prev_detection_frame_list if item[-2] == obj_id), None)
                if prev_detection:
                    if prev_detection[1] < box[1] and prev_detection[3] < box[3]:
                        direction = "F"
                    elif prev_detection[1] > box[1] and prev_detection[3] > box[3]:
                        direction = "B"
                    
        return direction
    
    
    # Lane violation detection
    def lane_violation_detection(self, obj_id, box, frame, frame_idx, direction):
        side, distance = "", 0
        x1, y1, x2, y2 = box

        if direction == "F":
            side, distance = self.get_side_of_movement((x1, np.mean([y1, y1])), self.lane_violation_line_coordinates[frame_idx])
        elif direction == "B":
            side, distance = self.get_side_of_movement((x2, np.mean([y2, y2])), self.lane_violation_line_coordinates[frame_idx])
        else:
            side, distance = self.get_side_of_movement((np.mean([x1,y1]), np.mean([y1, y1])), self.lane_violation_line_coordinates[frame_idx])

        if direction == "F" and side == "L":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, f"lane violation", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            violation_info_for_id =  self.lane_violation_object_info[frame_idx].get(obj_id, None)
            if violation_info_for_id:
                self.lane_violation_object_info[frame_idx][obj_id].append(self.frame_count)
            else:
                self.lane_violation_object_info[frame_idx][obj_id] = [self.frame_count]

            # Write the violation video chunk
            if obj_id not in self.violation_video_chunk_writers[frame_idx].keys():
                self.violation_video_chunk_writers[frame_idx][obj_id] = cv2.VideoWriter(
                    f"output/violation_video_chunks/{frame_idx}/{obj_id} {datetime.datetime.now().strftime('%H-%M-%S')} lane-violation.mp4", 
                    self.fourcc, 
                    int(self.sample_fps / self.escape_frame), 
                    (frame.shape[1], frame.shape[0])
                )
            self.violation_video_chunk_writers[frame_idx][obj_id].write(frame)
        
        elif direction == "B" and side == "R":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, f"lane violation", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            violation_info_for_id =  self.lane_violation_object_info[frame_idx].get(obj_id, None)
            if violation_info_for_id:
                self.lane_violation_object_info[frame_idx][obj_id].append(self.frame_count)
            else:
                self.lane_violation_object_info[frame_idx][obj_id] = [self.frame_count]

            # Write the violation video chunk
            if obj_id not in self.violation_video_chunk_writers[frame_idx].keys():
                self.violation_video_chunk_writers[frame_idx][obj_id] = cv2.VideoWriter(
                    f"output/violation_video_chunks/{frame_idx}/{obj_id} {datetime.datetime.now().strftime('%H-%M-%S')} lane-violation.mp4", 
                    self.fourcc, 
                    int(self.sample_fps / self.escape_frame), 
                    (frame.shape[1], frame.shape[0])
                )
            self.violation_video_chunk_writers[frame_idx][obj_id].write(frame)

        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            # cv2.putText(frame, f"", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


    # Red light violation detection
    def red_light_violation_detection(self, frame, frame_idx, obj_id, box, direction):
        if direction == "F" and box[2] >= self.red_light_violation_line_coordinates[frame_idx][0] + self.violation_allowed_limit_px and box[3] >= self.red_light_violation_line_coordinates[frame_idx][1] + self.violation_allowed_limit_px:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
            cv2.putText(frame, f"red light violation", (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if obj_id not in self.red_light_violation_object_ids[frame_idx]:
                self.red_light_violation_object_ids[frame_idx].append(obj_id)
                cv2.imwrite(f"output/red_light_violation/{frame_idx}/{obj_id} {datetime.datetime.now().strftime('%H-%M-%S')}.jpg", frame)
                with open(f"output/data/{frame_idx}/red_light_violation.txt", "a") as f:
                    f.write(f"camera-{frame_idx} {obj_id} {datetime.datetime.now().strftime('%H-%M-%S')}\n")

            # Write the violation video chunk
            if obj_id not in self.violation_video_chunk_writers[frame_idx].keys():
                self.violation_video_chunk_writers[frame_idx][obj_id] = cv2.VideoWriter(
                    f"output/violation_video_chunks/{frame_idx}/{obj_id} {datetime.datetime.now().strftime('%H-%M-%S')} red-light-violation.mp4", 
                    self.fourcc, 
                    int(self.sample_fps / self.escape_frame), 
                    (frame.shape[1], frame.shape[0])
                )
            self.violation_video_chunk_writers[frame_idx][obj_id].write(frame)


    # Parking violation detection
    def parking_violation_detection(self, frame, frame_idx, obj_id, box):
        if len(self.prev_detections) >= self.save_length -2:
            prev_detection_all = self.prev_detections[0]
            if len(prev_detection_all) > frame_idx:
                prev_detection_frame = prev_detection_all[frame_idx]

                prev_detection = next((item for item in prev_detection_frame if item[-2] == obj_id), None)
                if prev_detection:
                    np.array_equal(prev_detection[:4], box)
                    voilation_info_for_id = self.parking_violation_object_info[frame_idx].get(obj_id, None)
                    if voilation_info_for_id:
                        self.parking_violation_object_info[frame_idx][obj_id].append(self.frame_count)
                    else:
                        self.parking_violation_object_info[frame_idx][obj_id] = [self.frame_count]
    
    def run(self):
        try:
            # Create separate windows for each frame
            if self.show_video_output:
                for idx, _ in enumerate(self.videos):
                    cv2.namedWindow(f"camera {idx}", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(f"camera {idx}", self.window_size[0], self.window_size[1])

            # Read the frames form every video and perform object detection and tracking continuously
            while True:
                # Read the frames from every video and generate frames list
                frames_list = []
                for idx, (cap, fps, width, height) in enumerate(self.videos):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Error: Unable to read frame from video at {self.video_paths[idx]}.")
                    frames_list.append(frame)

                # Skip frames
                self.frame_count += 1
                if self.frame_count % self.escape_frame != 0:
                    continue

                # Add frames to prev frames
                self.prev_frames.append(frames_list)
                if len(self.prev_frames) > self.save_length:
                    self.prev_frames.pop(0)

                # Initialize video chunks writers
                if self.frame_count % (self.sample_fps * self.video_chunk_size) == 0:
                    if self.video_chunk_writers:
                        for video_chunk_writer in self.video_chunk_writers:
                            video_chunk_writer.release()
                    self.video_chunk_writers = None

                if not self.video_chunk_writers:
                    self.video_chunk_writers = [
                        cv2.VideoWriter(
                            f"output/video_chunks/{idx}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4", 
                            cv2.VideoWriter_fourcc(*"mp4v"), 
                            int(self.sample_fps / self.escape_frame), 
                            (width, height)
                        ) for idx, (cap, fps, width, height) in enumerate(self.videos)]
                    
                # Draw the lines if required
                if self.draw_lines:
                    for idx, frame in enumerate(frames_list):
                        cv2.line(frame, self.lane_violation_line_coordinates[idx][0], self.lane_violation_line_coordinates[idx][1], (0, 255, 0), 2)
                        if self.red_lights[idx]:
                            cv2.line(frame, self.red_light_violation_line_coordinates[idx][0], self.red_light_violation_line_coordinates[idx][1], (0, 0, 255), 2)

                # Perform object detection
                detections = self.model.predict(frames_list, stream=True, conf=self.conf, classes=self.track_classes)
                detections = list(detections)

                # If objects are detected then perform object tracking
                if len(detections) > 0:
                    temp_all_frame_prev_detections = []

                    # Clean the detections
                    for idx, detection in enumerate(detections):
                        xywhs = detection.boxes.xywh.cpu().numpy()
                        classes = detection.boxes.cls.cpu().numpy()
                        confs = detection.boxes.conf.cpu().numpy()

                        # Track objects using deep sort
                        trackings = self.deepsorts[idx].update(xywhs, confs, classes, frames_list[idx])

                        # If objects are tracked then perform further operations
                        if len(trackings) > 0:
                            bboxes_xyxy = trackings[:, :4]
                            object_ids = trackings[:, -2]
                            classes = trackings[:, -1]

                            temp_single_frame_prev_detections = []

                            # For every object tracked perform further operations
                            for bbox_xyxy, object_id, class_id in zip(bboxes_xyxy, object_ids, classes):
                                x1, y1, x2, y2 = [int(i) for i in bbox_xyxy]
                                temp_single_frame_prev_detections.append([x1, y1, x2, y2, object_id, class_id])

                                # Get the direction of movement of the object
                                direction = self.get_direction_of_movement([x1, y1, x2, y2], object_id, idx)

                                # Check if the object is violating the lane
                                if self.detect_lane_violation:
                                    self.lane_violation_detection(object_id, [x1, y1, x2, y2], frames_list[idx], idx, direction)

                                # Check if the object is violating the red light
                                if self.detect_red_light_violation and self.red_lights[idx]:
                                    self.red_light_violation_detection(frames_list[idx], idx, object_id, [x1, y1, x2, y2], direction)

                                # Check if the object is violating the parking
                                if self.detect_parking_violation and (self.frame_count % self.sample_fps) == 0:
                                    self.parking_violation_detection(frames_list[idx], idx, object_id, [x1, y1, x2, y2])

                            temp_all_frame_prev_detections.append(temp_single_frame_prev_detections)
                    
                    # Add detections to prev detections and remove the oldest detections
                    self.prev_detections.append(temp_all_frame_prev_detections)
                    if len(self.prev_detections) > self.save_length:
                        self.prev_detections.pop(0)

                    # Remove unwanted informations from lane_violation_object_info and write lane violation data and image
                    deleatable_keys = []
                    for key in self.lane_violation_object_info.keys():
                        for obj_id, frames in self.lane_violation_object_info[key].items():
                            if len(frames) < 2:
                                continue
                            elif frames[-1] - frames[-2] > self.escape_frame:
                                deleatable_keys.append(object_id)
                            elif len(frames) > self.allowed_violation_frame_count:
                                if obj_id not in self.lane_violation_object_ids[key]:
                                    self.lane_violation_object_ids[key].append(obj_id)
                                    deleatable_keys.append(object_id)
                                    cv2.imwrite(f"output/lane_violation/{key}/{obj_id} {datetime.datetime.now().strftime('%H-%M-%S')}.jpg", frames_list[key])
                                    with open(f"output/data/{key}/lane_violation.txt", "a") as f:
                                        f.write(f"camera-{key} {obj_id} {datetime.datetime.now().strftime('%H-%M-%S')}\n")

                    for key, value in self.lane_violation_object_info.items():
                        for deleatable_key in deleatable_keys:
                            if deleatable_key in value.keys():
                                del value[deleatable_key]

                    # Remove unwanted informations from parking_violation_object_info and write parking violation data and image
                    deleatable_keys = []
                    for key in self.parking_violation_object_info.keys():
                        for obj_id, frames in self.parking_violation_object_info[key].items():
                            if len(frames) > self.allowed_violation_frame_count:
                                if obj_id not in self.parking_violation_object_ids:
                                    self.parking_violation_object_ids.append(obj_id)
                                    deleatable_keys.append(object_id)
                                    cv2.imwrite(f"output/parking_violation/{key}/{obj_id} {datetime.datetime.now().strftime('%H-%M-%S')}.jpg", frames_list[key])
                                    with open(f"output/data/{key}/parking_violation.txt", "a") as f:
                                        f.write(f"camera-{key} {obj_id} {datetime.datetime.now().strftime('%H-%M-%S')}\n")

                    for key, value in self.parking_violation_object_info.items():
                        for deleatable_key in deleatable_keys:
                            if deleatable_key in value.keys():
                                del value[deleatable_key]

                # Check the deleatable ids for violation video chunk writers and release the writers
                if len(self.prev_detections) > self.save_length:
                    deleatable_ids = []
                    for prev_detection in self.prev_detections[0]:
                        if len(prev_detection) > 0:
                            for detection in prev_detection:
                                if detection[-2] not in deleatable_ids:
                                    deleatable_ids.append(detection[-2])

                    for frame_idx, video_chunk_writer in enumerate(self.violation_video_chunk_writers):
                        for obj_id in deleatable_ids:
                            if obj_id in video_chunk_writer.keys():
                                video_chunk_writer[obj_id].release()
                                del video_chunk_writer[obj_id]

                # Write the video chunks
                for idx, frame in enumerate(frames_list):
                    self.video_chunk_writers[idx].write(frame)

                # Show output video
                if self.show_video_output:
                    for idx, frame in enumerate(frames_list):
                        cv2.imshow(f"camera {idx}", frame)

                # Break the loop if no frames are found or quit if q is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # except Exception as e:
        #     print(e)
                
        finally:
            print(self.frame_count)

    
    def __del__(self):
        for video in self.videos:
            video.release()
        cv2.destroyAllWindows()         



if __name__ == "__main__":
    video_paths = [ "videos/traffic_vid_3.mp4"]
    model_name = "yolo_models/yolov8n.pt"
    lane_violation_line_coordinates = [((200, 1), (65, 845))]
    red_light_violation_line_coordinates = [(1, 845)]

    traffic_violation_detection = TrafficViolationDetection(
        video_paths,
        model_name,
        lane_violation_line_coordinates,
        red_light_violation_line_coordinates,
    )
    traffic_violation_detection.run()
