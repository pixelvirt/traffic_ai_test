from ultralytics import YOLO


model = YOLO("yolo_models/yolov8l.pt")
model.export(format="saved_model")
