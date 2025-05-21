import socket
import keyboard  
import time
import struct

# OpenBCI GUI's IP and Port (default for UDP is 127.0.0.1:12345)
UDP_IP = "127.0.0.1"
UDP_PORT = 12350

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("""
Marker Guide:
    b: blink (1)
    l: look left (2)
    r: look right (3)
    c: clench jaw (4)
    u: look up (5)
    d: look down (6)
    f: frustrated emotions (7)
    h: happy emotions (8)
Press 'q' to exit.
""")

actions = {
    "b": (1.0, "Blink marked!"),
    "l": (2.0, "Left look marked!"),
    "r": (3.0, "Right look marked!"),
    "c": (4.0, "Jaw clench marked!"),
    "u": (5.0, "Up look marked!"),
    "d": (6.0, "Down look marked!"),
    "f": (7.0, "Frustrated emotions marked!"),
    "h": (8.0, "Happy emotions marked!"),
}

while True:
    for key, (value, message) in actions.items():
        if keyboard.is_pressed(key):
            message_data = struct.pack("!f", value)
            sock.sendto(message_data, (UDP_IP, UDP_PORT))
            print(message)
            time.sleep(1)
            break  
    if keyboard.is_pressed("q"):  # Press 'q' to quit the script
        print("Exiting...")
        break

sock.close()
