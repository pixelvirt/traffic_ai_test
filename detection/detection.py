from typing import List
from numpy.typing import NDArray


from supervision import Detections
from ultralytics import YOLO
from video_source import VideoSource


class Detection:
    def __init__(self, model: str = "yolo_models/yolov8n.pt"):
        self.model = YOLO(model)
        self.model.fuse()

    @property
    def model_class_names(self):
        return self.model.model.names

    def countObjectsInFrame(self, frame) -> dict:
        results = self.model(frame)

        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )

        # Object counts
        objects = {}

        for cls in detections.class_id:
            cls_name = self.model_class_names[cls]
            if cls_name not in objects:
                objects[cls_name] = 0
            objects[cls_name] += 1

        return objects

    def countObjectsInFrames(self, frames: List[NDArray]) -> List[dict]:
        results = self.model(frames, verbose=False)

        objects = []

        for result in results:
            detections = Detections(
                xyxy=result.boxes.xyxy.cpu().numpy(),
                confidence=result.boxes.conf.cpu().numpy(),
                class_id=result.boxes.cls.cpu().numpy().astype(int),
            )

            # Object counts
            obj = {}

            for cls in detections.class_id:
                vehicle_classes = ["bicycle", "car", "motorcycle", "bus", "truck"]
                cls_name = self.model_class_names[cls]

                if cls_name not in vehicle_classes:
                    continue

                if cls_name not in obj:
                    obj[cls_name] = 0
                obj[cls_name] += 1
            obj["vehicles"] = sum(obj.values())
            objects.append(obj)

        return objects


if __name__ == "__main__":
    detection = Detection(model="yolo_models/yolov8l.pt")

    # Source video path
    SOURCE_VIDEO_PATH = "videos/traffic.mp4"
    SOURCE_VIDEO_PATH2 = "videos/chabahil.mp4"

    # Clear detections.txt
    with open("detections.txt", "w") as f:
        f.write("")

    videos = [
        VideoSource(SOURCE_VIDEO_PATH),
        VideoSource(SOURCE_VIDEO_PATH2),
        VideoSource(SOURCE_VIDEO_PATH),
        VideoSource(SOURCE_VIDEO_PATH2),
        VideoSource(SOURCE_VIDEO_PATH),
        VideoSource(SOURCE_VIDEO_PATH2),
        VideoSource(SOURCE_VIDEO_PATH),
        VideoSource(SOURCE_VIDEO_PATH2),
    ]

    frames = [video.frames for video in videos]

    count = 0
    while True:
        try:
            temp_frames = [next(frame) for frame in frames]

            count += 1

            if count % 30 != 0:
                continue

            print(f"Processing batch of {len(frames)} frames")
            print("Current frame at:", count)
            objects = detection.countObjectsInFrames(temp_frames)
            print("Writing to detections.txt")
            with open("detections.txt", "a") as f:
                for index, camera in enumerate(objects):
                    f.write(f"Camera {index + 1}: {camera}\n")
                f.write("\n")
            print("Written to detections.txt")
        except StopIteration:
            break
print("Completed")
