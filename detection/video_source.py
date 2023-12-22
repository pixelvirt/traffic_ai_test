from supervision import VideoInfo
from supervision import get_video_frames_generator


class VideoSource:
    def __init__(self, source_path: str):
        self.source = source_path

    @property
    def video_info(self):
        return VideoInfo.from_video_path(self.source)

    @property
    def frames(self):
        return get_video_frames_generator(self.source)

    def __repr__(self):
        return f"VideoSource(source={self.source})"


if __name__ == "__main__":
    # Source video path
    SOURCE_VIDEO_PATH = "traffic.mp4"

    # create frame generator
    video = VideoSource(SOURCE_VIDEO_PATH)

    for frame in video.frames:
        print(frame.shape)
        break
