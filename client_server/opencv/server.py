import socket
import threading

import cv2

HOST = socket.gethostbyname(socket.gethostname())
PORT = 9080

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))

clients = {}


def start_traffic_video_processing():
    ip_address = "192.168.101.21"
    port = 8080
    ip_camera_url = f"http://{ip_address}:{port}/video"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    while True:
        frames_list = []
        success, img_1 = cap.read()
        frames_list.append(img_1)
        success, img_2 = cap.read()
        frames_list.append(img_2)
        
        if not success:
            print("Unable to read frame from camera")
            break

        for addr, client in list(clients.items()):
            try:
                client.send(str(len(frames_list)).encode("utf-8").ljust(64))
                for frame in frames_list:
                    _, jpeg = cv2.imencode('.jpg', frame)
                    img_bytes = jpeg.tobytes()

                    data_len = len(img_bytes)
                    client.send(str(data_len).encode("utf-8").ljust(64))
                    client.send(img_bytes)
            except BrokenPipeError:
                print(f"Connection with {addr} is broken. Removing from clients.")
                client.close()
                clients.pop(addr)

    cap.release()


def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    while connected:
        msg_length = conn.recv(64).decode("utf-8")
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode("utf-8")
            if msg == "quit":
                clients.pop(addr)
                connected = False
            print(f"[{addr}] {msg}")
            conn.send("Msg received".encode("utf-8"))

    conn.close()


def start_listening_to_client():
    print("Server is starting...")
    server.listen()
    print(f"[LISTENING] Server is listening on {HOST}:{PORT}")
    while True:
        conn, addr = server.accept()
        print(conn, addr)
        clients[addr] = conn
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.active_count() - 3}")


traffic_thread = threading.Thread(target=start_traffic_video_processing)
traffic_thread.start()

listening_thread = threading.Thread(target=start_listening_to_client)
listening_thread.start()

traffic_thread.join()
listening_thread.join()
