import cv2
import numpy as np
import subprocess

def capture_and_encode(video_source=0, output_playlist='output/output.m3u8', output_segment_prefix='output_segment', segment_duration=5):
    cap = cv2.VideoCapture(video_source)
    
    # Set video width and height according to your preferences
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Output video settings
    output_fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # FFmpeg command to create the playlist
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # overwrite output files without asking
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{output_resolution[0]}x{output_resolution[1]}',
        '-pix_fmt', 'bgr24',
        '-r', str(output_fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-threads', '1',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-pix_fmt', 'yuv420p',
        '-b:v', '500k',
        '-g', str(output_fps * segment_duration),
        '-hls_time', str(segment_duration),
        '-hls_list_size', '5',
        '-hls_segment_filename', f'output/{output_segment_prefix}_%03d.ts',
        output_playlist,
    ]

    process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # You can perform image processing or other modifications here if needed

            process.stdin.write(frame.tobytes())

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        process.stdin.close()
        process.wait()

if __name__ == "__main__":
    capture_and_encode()
