import cv2

ip_address = "192.168.101.19"
port = 8080

ip_camera_url = f"http://{ip_address}:{port}/video"


cap1 = cv2.VideoCapture(ip_camera_url)

cap2 = cv2.VideoCapture(0)

while True:
    ret, frame1 = cap1.read()
    rotated_frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
    resized_frame1 = cv2.resize(rotated_frame1, (640, 480))

    ret, frame2 = cap2.read()
    resized_frame2 = cv2.resize(frame2, (640, 480))

    frame = cv2.hconcat([resized_frame1, resized_frame2])
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap1.release()
cv2.destroyAllWindows()
