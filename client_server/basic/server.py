import socket
import threading


HOST = socket.gethostbyname(socket.gethostname())
PORT = 9080

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))


def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    while connected:
        msg_length = conn.recv(64).decode("utf-8")
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode("utf-8")
            if msg == "quit":
                connected = False
            print(f"[{addr}] {msg}")
            conn.send("Msg received".encode("utf-8"))

    conn.close()


def start():
    server.listen()
    print(f"[LISTENING] Server is listening on {HOST}:{PORT}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.active_count() - 1}")


print("Server is starting...")
start()
