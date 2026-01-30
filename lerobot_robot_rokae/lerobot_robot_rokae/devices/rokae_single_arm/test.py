# state_listener.py
import zmq
import threading
from rokae_python_wrapper.rokae_server import RobotState
import requests
import time


def state_listener():
    ctx = zmq.Context.instance()

    state_socket = ctx.socket(zmq.SUB)
    state_socket.connect("tcp://localhost:5556")

    # 订阅所有 topic（必须有）
    state_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    print("[StateListener] started")
    try:
        while True:
            state = state_socket.recv_pyobj()
            print("[StateListener] state:", state)
    except KeyboardInterrupt:
        print("Ctrl+C received, shutting down...")
    finally:
        try:
            # self.cmd_socket.close(0)
            state_socket.close(0)
        except Exception as e:
            print("Socket close error:", e)
        try:
            ctx.term()
        except Exception as e:
            print("Context term error:", e)

BASE_URL = "http://127.0.0.1:5000"

def is_in_realtime_loop():
    url = f"{BASE_URL}/is_in_realtime_loop"
    r = requests.get(url)
    return r.json()

def start_realtime_loop():
    url = f"{BASE_URL}/start_realtime_loop"
    payload = {"rt_control_mode": "cartesian_position", "callback_mode":"cart_pos"}
    r = requests.post(url, json = payload)
    return r.json()

def get_state(quantities):
    url = f"{BASE_URL}/get_state"
    r = requests.get(url, params={
        "quantities": ",".join(quantities)
    })
    r.raise_for_status()
    return r.json()

def open_gripper():
    url = f"{BASE_URL}/open_gripper"
    r = requests.post(url)
    return r.json()

def close_gripper():
    url = f"{BASE_URL}/close_gripper"
    r = requests.post(url)
    return r.json()

if __name__ == "__main__":
    # state_listener()
    # start_realtime_loop()
    # while True:
    #     print(f"is in real time loop = {is_in_realtime_loop()}")
    #     state = get_state(["joint_pos_cmd", "cart_pos_cmd"])
    #     print(state)
    #     # print(f"joint_pos_cmd = {state['joint_pos_cmd']}, cart_pos_cmd = {state['cart_pos_cmd']}")
    #     time.sleep(2)
    open_gripper()
    time.sleep(2)
    close_gripper()
    time.sleep(2)
    open_gripper()
    time.sleep(2)
    close_gripper()
    time.sleep(2)
    open_gripper()
    time.sleep(2)
    close_gripper()
    time.sleep(2)
    open_gripper()
    time.sleep(2)
    close_gripper()
    time.sleep(2)
