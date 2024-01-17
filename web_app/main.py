from flask import Flask, render_template, Response
import cv2
import numpy as np
from example import main, create_output_folders, get_video_info, initialize_deepsort, get_model, device

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html') 

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""

    # Main Logic
    model_name = "yolo_models/yolov8n.pt"
    model = get_model(model_name, device)

    # video_paths = ["videos/traffic_vid_3.mp4", "videos/traffic_vid_2.mp4", "videos/traffic_vid_3.mp4"]
    video_paths = ["videos/traffic_vid_3.mp4", "videos/traffic_vid_2.mp4"]
    deepsort = [initialize_deepsort() for _ in range(len(video_paths))]
    cap_list = [get_video_info(video_path) for video_path in video_paths]

    for idx, cap in enumerate(cap_list):
        if not cap[0].isOpened():
            raise ValueError(f"Error: Unable to open video {idx}.")

    # mid_line_coordinates = [((200, 1), (65, 845)), ((260,1), (1,625)), ((200, 1), (65, 845))]
    # red_line_coordinates = [(80,790), (1, 650), (80,790)]
    mid_line_coordinates = [((200, 1), (65, 845)), ((260,1), (1,625))]
    red_line_coordinates = [(80,790), (1, 650)]
    check_parking_violation = True
    check_red_light_violation = True
    red_light = False
    escape_frame = 2
    violation_frame = 5
    violation_allowed_limit_px = 10    # pixel from the line up to which the violation is allowed
    num_prev_positions = 5              # number of previous positions to consider for detection of parking violation

    if not len(mid_line_coordinates) == len(cap_list) or not len(red_line_coordinates) == len(cap_list):
        raise ValueError("Error: Line coordinates and video count must be equal.")

    create_output_folders()

    return Response(
    main(
        cap_list, 
        model, 
        deepsort, 
        mid_line_coordinates, 
        red_line_coordinates, 
        violation_allowed_limit_px, 
        escape_frame, violation_frame, 
        check_red_light_violation, 
        check_parking_violation, 
        num_prev_positions
        ),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port =5000, debug=True, threaded=True)