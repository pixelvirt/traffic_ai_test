import os
import datetime
import subprocess

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config

from coco_names import coco_class_list


class TrafficViolationDetection:
    def __init__(
        self,                                       # represents the current instance of the class
        camera_ids,                                 # list of ids for every video
        video_paths,                                # list of paths for traffic violation detection (may be list of location of files or camera urls)
        model_name,                                 # name of the yolo model to be used for detection
        lane_violation_line_coordinates,            # coordinates of the line for lane violation detection
        red_light_violation_line_coordinates,       # coordinates of the line for red light violation detection
        output_dir,
        escape_frame = 2,                           # number of frames to skip
        allowed_violation_frame_count = 5,          # number of frames allowed for violation
        lane_violation_allowed_limit_px = 10,       # number of pixels allowed for violation in case of lane violation light violation
        red_light_violation_allowed_limit_px = 25,  # number of pixels allowed for violation in case of red light violation light violation
    ):
        # Initialize videos from video paths and check if the provided video paths are working or not  
        self.camera_ids = camera_ids
        self.video_paths = video_paths
        self.video_info = {camera_id: self.get_video_info(video_path) for camera_id, video_path in self.video_paths.items()}

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
        self.deepsorts = {camera_id: self.initialize_deepsort() for camera_id in self.camera_ids}

        # Load the coco class list
        self.classes_list = coco_class_list

        # Initialize the lane violation line coordinates
        self.lane_violation_line_coordinates = lane_violation_line_coordinates
        self.red_light_violation_line_coordinates = red_light_violation_line_coordinates
        if not len(self.lane_violation_line_coordinates) == len(self.red_light_violation_line_coordinates) == len(self.camera_ids):
            raise ValueError("Error: lane_violation_line_coordinates and  red_light_violation_line_coordinates must be provided for every video.")
        
        # Initialize red light for every video
        self.red_lights = {camera_id: False for camera_id in self.camera_ids}

        # Initialize the type of violation to detect
        self.detect_lane_violation = True
        self.detect_red_light_violation = True
        self.detect_parking_violation = True

        # Initialize the escape frame and allowed violation frame
        self.escape_frame = escape_frame
        self.allowed_violation_frame_count = allowed_violation_frame_count
        self.lane_violation_allowed_limit_px = lane_violation_allowed_limit_px
        self.red_light_violation_allowed_limit_px = red_light_violation_allowed_limit_px

        # Initialize the prev frames and prev detections
        self.save_length = 60       # number of frames and detections to save (edit this if necessary)
        self.prev_frames = []
        self.prev_detections = []
        self.temp_prev_detection = {}

        # Initialize violation data
        self.lane_violation_object_ids = {camera_id: [] for camera_id in self.camera_ids}
        self.lane_violation_object_info = {camera_id: {} for camera_id in self.camera_ids}
        self.red_light_violation_object_ids = {camera_id: [] for camera_id in self.camera_ids}
        self.parking_violation_object_ids = {camera_id: [] for camera_id in self.camera_ids}
        self.parking_violation_object_info = {camera_id: {} for camera_id in self.camera_ids}

        # Some extra variables
        self.frame_count = 0
        self.sample_fps = 30
        self.draw_lines = True
        self.show_video_output = True
        self.window_size = (720, 1280)

        self.output_dir = output_dir
        self.create_output_folders()

        # Initialize video chunk writers for every video and violations
        self.video_chunk_writers = {}
        self.video_chunk_size = 5
        self.video_writer_process = {
            camera_id: subprocess.Popen(self.get_ffmpeg_command(camera_id), stdin=subprocess.PIPE) for camera_id in self.camera_ids
        }



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
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        if not os.path.exists(os.path.join(self.output_dir, "violation_data")):
            os.mkdir(os.path.join(self.output_dir, "violation_data"))
        if not os.path.exists(os.path.join(self.output_dir, "video_chunks")):
            os.mkdir(os.path.join(self.output_dir, "video_chunks"))

        # Create output folders for every video camera
        for camera_id in self.camera_ids:
            if not os.path.exists(os.path.join(self.output_dir, f"violation_data/{camera_id}")):
                os.mkdir(os.path.join(self.output_dir, f"violation_data/{camera_id}"))
            if not os.path.exists(os.path.join(self.output_dir, f"video_chunks/{camera_id}")):
                os.mkdir(os.path.join(self.output_dir, f"video_chunks/{camera_id}"))


    # Set the lane violation line coordinates in case changes are required
    def set_lane_violation_line_coordinates(self, lane_violation_line_coordinates, camera_id):
        self.lane_violation_line_coordinates[camera_id] = lane_violation_line_coordinates


    # Set the red light violation line coordinates in case changes are required
    def set_red_light_violation_line_coordinates(self, red_light_violation_line_coordinates, camera_id):
        self.red_light_violation_line_coordinates[camera_id] = red_light_violation_line_coordinates


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
        if distance >= self.lane_violation_allowed_limit_px:
            side = "R"
        elif distance <= -(self.lane_violation_allowed_limit_px):
            side = "L"
        else:
            side = "N"

        return side, distance
    

    # Get the direction of movement of an object (forward, backward or none)
    def get_direction_of_movement(self, box, obj_id, camera_id):
        direction = "N"

        previous_detections_length = len(self.prev_detections)
        if previous_detections_length > 0:
            compare_index = min(previous_detections_length, 3)
            previous_detection = self.prev_detections[-compare_index]
            if not previous_detection:
                return direction
            previous_detection_for_id = next((item for item in previous_detection[camera_id] if item[-2] == obj_id), None)
            if previous_detection_for_id:
                if np.mean([previous_detection_for_id[1], previous_detection_for_id[3]]) > np.mean([box[1], box[3]]):
                    direction = "B"
                elif np.mean([previous_detection_for_id[1], previous_detection_for_id[3]]) < np.mean([box[1], box[3]]):
                    direction = "F"
                    
        return direction
    

    # Lane violation detection
    def lane_violation_detection(self, object_id, class_id, box, frame, camera_id, direction):
        side, distance = "N", 0
        x1, y1, x2, y2 = box

        if class_id in [1, 3]:
            if direction == "F":
                side, distance = self.get_side_of_movement([x2, np.mean([y1, y2])], self.lane_violation_line_coordinates[camera_id])
            elif direction == "B":
                side, distance = self.get_side_of_movement([x1, np.mean([y1, y2])], self.lane_violation_line_coordinates[camera_id])
            else:
                side, distance = self.get_side_of_movement([np.mean([x1, x2]), np.mean([y1, y2])], self.lane_violation_line_coordinates[camera_id])
        else:
            side, distance = self.get_side_of_movement([np.mean([x1, x2]), np.mean([y1, y2])], self.lane_violation_line_coordinates[camera_id])

        
        if direction == "F" and side == "L":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, f"lane violation", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if object_id not in self.lane_violation_object_ids[camera_id]:
                violation_info_for_id =  self.lane_violation_object_info[camera_id].get(object_id, None)
                if violation_info_for_id:
                    self.lane_violation_object_info[camera_id][object_id].append(self.frame_count)
                else:
                    self.lane_violation_object_info[camera_id][object_id] = [self.frame_count]
        
        elif direction == "B" and side == "R":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, f"lane violation", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if object_id not in self.lane_violation_object_ids[camera_id]:
                violation_info_for_id =  self.lane_violation_object_info[camera_id].get(object_id, None)
                if violation_info_for_id:
                    self.lane_violation_object_info[camera_id][object_id].append(self.frame_count)
                else:
                    self.lane_violation_object_info[camera_id][object_id] = [self.frame_count]

        # else:
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            # cv2.putText(frame, f"{direction} {side}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


    # Red light violation detection
    def red_light_violation_detection(self, frame, camera_id, object_id, box, direction):
        if direction == "F" and np.mean([box[1], box[3]]) > self.red_light_violation_line_coordinates[camera_id][0][1] + self.red_light_violation_allowed_limit_px:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(frame, "Red Light Violation", (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if object_id not in self.red_light_violation_object_ids[camera_id]:
                self.red_light_violation_object_ids[camera_id].append(object_id)
                self.write_violation_data(camera_id, object_id, "red_light_violation")
    

    # Write the violation data
    def write_violation_data(self, camera_id, object_id, violation_type):
        length_to_write = min(len(self.prev_frames) - 2, (self.sample_fps * 3) - 2)
        crop_length = len(self.prev_frames) - length_to_write

        image_frames = [frame_dict[camera_id] for frame_dict in self.prev_frames[crop_length:] if frame_dict.get(camera_id, None) is not None]
        image_detections = [detection_info[camera_id] for detection_info in self.prev_detections[crop_length:] if detection_info.get(camera_id, None) is not None]

        time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        
        img_frame = image_frames[-1]
        box = next((item for item in image_detections[-1] if item[-2] == object_id), None)
        image_path = f"{self.output_dir}/violation_data/{camera_id}/{object_id} {violation_type} {time}.jpg"
        cv2.circle(img_frame, (int(np.mean([box[0], box[1]])), int(np.mean([box[2], box[3]]))), 3, (0, 0, 255), -1)
        cv2.imwrite(image_path, img_frame)

        video_path = f"{self.output_dir}/violation_data/{camera_id}/{object_id} {violation_type} {time}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_path, fourcc, int(self.sample_fps / self.escape_frame), (img_frame.shape[1], img_frame.shape[0]))

        for frame, detection in zip(image_frames, image_detections):
            detection_for_id = next((item for item in detection if item[-2] == object_id), None)
            if detection_for_id:
                box = detection_for_id[:4]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
                cv2.circle(frame, (int(np.mean([box[0], box[1]])), int(np.mean([box[2], box[3]]))), 3, (0, 0, 255), -1)
                # cv2.putText(frame, f"{violation_type}", (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            video_writer.write(frame)

        video_writer.release()


    def get_ffmpeg_command(self, camera_id):
        return [
            'ffmpeg',
            '-y',  # overwrite output files without asking
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.video_info[camera_id][2]}x{self.video_info[camera_id][3]}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.video_info[camera_id][1]),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p',
            '-b:v', '500k',
            '-g', str(self.video_info[camera_id][1] * self.video_chunk_size),
            '-hls_time', str(self.video_chunk_size),
            '-hls_list_size', '5',
            '-hls_segment_filename', f'{self.output_dir}/video_chunks/{camera_id}/video_chunk_%03d.ts',
            f'{self.output_dir}/video_chunks/{camera_id}/output.m3u8'
        ]

    
    # Sart the violation detection process
    def run(self):
        try:
            # Create separate windows for each frame
            if self.show_video_output:
                for camera_id in self.camera_ids:
                    cv2.namedWindow(f"camera {camera_id}", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(f"camera {camera_id}", self.window_size[0], self.window_size[1])


            # Read the frames form every video and perform object detection and tracking continuously
            while True:
                try:

                    # Read the frames from every video and generate frames dict
                    frames_dict = {}
                    for camera_id, (cap, _, _, _) in self.video_info.items():
                        ret, frame = cap.read()
                        if not ret:
                            print(f"Error: Unable to read frame from video for camera id {camera_id}.")
                        frames_dict[camera_id] = frame

                    # Reset frame count to 0 for better performance
                    if self.frame_count == 2400:
                        self.frame_count = 0

                    # Skip frames
                    self.frame_count += 1
                    if self.frame_count % self.escape_frame != 0:
                        continue

                    # Add frames to prev frames
                    self.prev_frames.append(frames_dict)
                    if len(self.prev_frames) > self.save_length:
                        self.prev_frames.pop(0)

                    # Draw the lines if required
                    if self.draw_lines:
                        for camera_id, frame in frames_dict.items():
                            if self.lane_violation_line_coordinates[camera_id]:
                                cv2.line(frame, self.lane_violation_line_coordinates[camera_id][0], self.lane_violation_line_coordinates[camera_id][1], (0, 255, 0), 2)
                            if self.red_lights[camera_id] and self.red_light_violation_line_coordinates[camera_id]:
                                cv2.line(frame, self.red_light_violation_line_coordinates[camera_id][0], self.red_light_violation_line_coordinates[camera_id][1], (0, 0, 255), 2)

                    # Prepare frameslist as yolo takes list not dictionary for prediction
                    detection_frames = [[], []]
                    for camera_id, frame in frames_dict.items():
                        detection_frames[0].append(camera_id)
                        detection_frames[1].append(frame)


                    # Perform object detection
                    detections = self.model.predict(detection_frames[1], stream=True, conf=self.conf, classes=self.track_classes, verbose=False)
                    detections = list(detections)

                    detection_info = {
                        camera_id: detection for camera_id, detection in zip(detection_frames[0], detections)
                    }

                    # If objects are detected then perform object tracking
                    if len(detections) > 0:
                        temp_prev_detections = {}

                        # Clean the detections
                        for camera_id, detection in detection_info.items():
                            xywhs = detection.boxes.xywh.cpu().numpy()
                            classes = detection.boxes.cls.cpu().numpy()
                            confs = detection.boxes.conf.cpu().numpy()

                            # Track objects using deep sort
                            trackings = self.deepsorts[camera_id].update(xywhs, confs, classes, frames_dict[camera_id])

                            # If objects are tracked then perform further operations
                            if len(trackings) > 0:
                                bboxes_xyxy = trackings[:, :4]
                                object_ids = trackings[:, -2]
                                classes = trackings[:, -1]

                                temp_prev_detections_for_camera = []

                                # For every object tracked perform further operations
                                for bbox_xyxy, object_id, class_id in zip(bboxes_xyxy, object_ids, classes):
                                    x1, y1, x2, y2 = [int(i) for i in bbox_xyxy]
                                    temp_prev_detections_for_camera.append([x1, y1, x2, y2, object_id, class_id])

                                    # Draw the bounding box for every object
                                    cv2.rectangle(frames_dict[camera_id], (x1, y1), (x2, y2), (255, 0, 0), 2)

                                    # Get the direction of movement of the object
                                    direction = self.get_direction_of_movement([x1, y1, x2, y2], object_id, camera_id)

                                    # Check if the object is violating the lane
                                    if self.detect_lane_violation and self.lane_violation_line_coordinates[camera_id]:
                                        self.lane_violation_detection(object_id, class_id, [x1, y1, x2, y2], frames_dict[camera_id], camera_id, direction)

                                    # Check if the object is violating the red light
                                    if self.detect_red_light_violation and self.red_light_violation_line_coordinates[camera_id] and self.red_lights[camera_id]:
                                        self.red_light_violation_detection(frames_dict[camera_id], camera_id, object_id, [x1, y1, x2, y2], direction)

                                temp_prev_detections[camera_id] = temp_prev_detections_for_camera

                        # Add detections to prev detections and remove the oldest detections
                        self.prev_detections.append(temp_prev_detections)
                        if len(self.prev_detections) > self.save_length:
                            self.prev_detections.pop(0)

                        # Remove unwanted informations from lane_violation_object_info and write lane violation data and image
                        deleatable_keys = {camera_id: [] for camera_id in self.camera_ids}
                        for camera_id, lane_violation_info in self.lane_violation_object_info.items():
                            for object_id, violation_frames_list in lane_violation_info.items():
                                if len(violation_frames_list) > 2:
                                    if violation_frames_list[-1] - violation_frames_list[-2] > self.escape_frame:
                                        deleatable_keys[camera_id].append(object_id)

                                    if len(violation_frames_list) > self.allowed_violation_frame_count:
                                        deleatable_keys[camera_id].append(object_id)
                                        if not object_id in self.lane_violation_object_ids[camera_id]:
                                            self.lane_violation_object_ids[camera_id].append(object_id)
                                            self.write_violation_data(camera_id, object_id, "lane_violation")

                        for camera_id, object_ids in deleatable_keys.items():
                            for object_id in self.parking_violation_object_ids[camera_id]:
                                del self.lane_violation_object_info[camera_id][object_id]
                    
                    # Write the video chunks using ffmpeg
                    for camera_id, frame in frames_dict.items():
                        self.video_writer_process[camera_id].stdin.write(frame.tobytes())

                    # Show the video output if required
                    if self.show_video_output:
                        for camera_id, frame in frames_dict.items():
                            cv2.imshow(f"camera {camera_id}", frame)

                    # Break the loop if no frames are found or quit if q is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                except Exception as e:
                    print(e)
                    for cap, _, _, _ in self.video_info.values():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        except Exception as e:
            print(e)
            
        finally:
            print(self.frame_count)


    # Release all video captures when object is destroyed
    def __del__(self):
        for cap, _, _, _ in self.video_info.values():
            cap.release()
        cv2.destroyAllWindows


def start_violation_detection():
    cameras = [{"id": 1, "source": "videos/traffic_vid_3.mp4"}, {"id": 2, "source": "videos/traffic_vid_3.mp4"}]
    coordinates = [
        {"id": 1, "lane_violation": ((200, 1), (65, 845)), "red_light_violation": ((80, 790), (470, 790))}, 
        {"id": 2, "lane_violation": ((200, 1), (65, 845)), "red_light_violation": ((80, 790), (470, 790))}
    ]

    if cameras:
        camera_ids = [ camera["id"] for camera in cameras ]
        video_paths = { camera["id"]: camera["source"] for camera in cameras  }
        lane_violation_line_coordinates = { camera["id"]: coordinate.get("lane_violation", None) for camera, coordinate in zip(cameras, coordinates)}
        red_light_violation_line_coordinates = { camera["id"]: coordinate.get("red_light_violation", None) for camera, coordinate in zip(cameras, coordinates)}
        output_dir = "output"
        model_name = f"yolo_models/yolov8n.pt"

        traffic_violation_detection = TrafficViolationDetection(
            camera_ids,
            video_paths,
            model_name,
            lane_violation_line_coordinates,
            red_light_violation_line_coordinates,
            output_dir,
        )
        traffic_violation_detection.run()


if __name__ == "__main__":
    start_violation_detection()
