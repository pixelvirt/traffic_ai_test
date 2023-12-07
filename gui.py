import tkinter
import cv2
import PIL.Image, PIL.ImageTk

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid_1 = MyVideoCapture(self.video_source)
        self.vid_2 = MyVideoCapture(self.video_source)

        self.window.geometry("1280x960")

        self.canvas_1 = tkinter.Canvas(window, width=640, height=480)
        self.canvas_1.grid(row=0, column=0)

        self.canvas_2 = tkinter.Canvas(window, width=640, height=480)
        self.canvas_2.grid(row=0, column=1)

        self.canvas_3 = tkinter.Canvas(window, width=640, height=480)
        self.canvas_3.grid(row=1, column=0)

        self.canvas_4 = tkinter.Canvas(window, width=640, height=480)
        self.canvas_4.grid(row=1, column=1)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid_1.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas_1.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")
