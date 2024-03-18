import socket

import cv2
import numpy as np


HOST = socket.gethostbyname(socket.gethostname())
PORT = 9080

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))


def send(msg):
    message = msg.encode("utf-8")
    msg_length = len(message)
    send_length = str(msg_length).encode("utf-8")
    send_length += b" " * (64 - len(send_length))
    client.send(send_length)
    client.send(message)


def receive_data():
    # Receive the length of the message
    msg_length = int(client.recv(64).decode("utf-8"))

    # Initialize an empty byte buffer
    data = b""

    # Continue receiving data until the entire message is received
    while len(data) < msg_length:
        remaining_bytes = msg_length - len(data)
        chunk_size = min(4096, remaining_bytes)  # Adjust the chunk size as needed
        chunk = client.recv(chunk_size)
        if not chunk:
            # Handle the case where the connection is closed unexpectedly
            print("Connection closed unexpectedly.")
            return None
        data += chunk

    return data


try:
    while True:
        num_frame = int(client.recv(64).decode("utf-8"))
        for i in range(num_frame):
            window_name = f"camera-{i}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            data = receive_data()
            if not data:
                print("No data received from server")
                break
            
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (640, 480))

            if img is None:
                continue

            cv2.imshow(f"camera-{i}", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            send("quit")
            break

finally:
    cv2.destroyAllWindows()
