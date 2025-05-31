import numpy as np
import time
import glob
import os
import sys
import carla
import cv2
import random

from config import CONFIG
from Control import CarlaControl
from YOLO import YOLODetector
from CarlaEnv import CarlaEnv
from CamManager import CamManager
from DQN import DQNAgent

state_size = 5
action_size = 7

ACTIONS = [
    (0.25, 0.0, 0.0, False),  # 直行加速
    (0.0, 0.0, 0.2, False),  # 直行減速
    (0.0, -0.6, 0.03, False),  # 左轉大
    (0.0, -0.4, 0.05, False),  # 左轉小
    (0.0, 0.6, 0.03, False),  # 右轉大
    (0.0, 0.4, 0.05, False),  # 右轉小
    (0.1, 0.0, 0.0, True),   # 倒車
]

def main():
    env = CarlaEnv()
    cam_manager = CamManager()
    # detector = YOLODetector()
    agent = DQNAgent(state_size, action_size, CONFIG['epsilon'], CONFIG['epsilon_min'], CONFIG['epsilon_decay'], CONFIG['target_update_freq'])
    agent.load("./model/dqn_ep2350.pth") # modify this for other paths
    
    try:
        env.spawn_vehicle()
        env.attach_camera(cam_manager.process_img)
        state = cam_manager.latest_frame
        print("\nPress Crtl + C to stop.\n")
        try:
            while True:
                if cam_manager.latest_frame is None:
                    continue
                # frame = cam_manager.latest_frame
                # detections = detector.detect(frame)
                action_index = agent.choose_action(state)
                action = ACTIONS[action_index]

                next_state, reward, done, _ = env.step(action, None)
                state = next_state
        except KeyboardInterrupt:
            print("\nExiting\n")
    finally:
        env.cleanup()

if __name__ == "__main__":
    main()