import cv2

ip_address = "192.168.101.19"
port = 8080

ip_camera_url = f"http://{ip_address}:{port}/video"


cap = cv2.VideoCapture(ip_camera_url)

while True:
    ret, frame = cap.read()
    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    resized_frame = cv2.resize(rotated_frame, (640, 480))
    cv2.imshow('frame', resized_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
