import socket


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


while True:
    msg = input("Enter your message: ")
    send(msg)
    if msg == "quit":
        send("quit")
        break
    print(client.recv(2048).decode("utf-8"))