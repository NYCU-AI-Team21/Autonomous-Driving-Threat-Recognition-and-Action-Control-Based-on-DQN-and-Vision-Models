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


def main():
    ACTIONS = [
        [0.5, 0.0, 0.0],   # go straight
        [0.5, -0.3, 0.0],  # turn left
        [0.5, 0.3, 0.0],   # turn right
        [0.0, 0.0, 1.0]    # brake
    ]
    state_size = 4
    action_size = 4
    
    env = CarlaEnv()
    cam_manager = CamManager()
    detector = YOLODetector()
    agent = DQNAgent(state_size, action_size, CONFIG['epsilon'], CONFIG['epsilon_min'], CONFIG['epsilon_decay'], CONFIG['target_update_freq'])
    agent.load("./model/dqn_ep490.pth") # modify this for other paths
    
    try:
        env.spawn_vehicle()
        env.attach_camera(cam_manager.process_img)
        state = [0.0, 0, 0, 0, 100.0]
        print("\nPress Crtl + C to stop.\n")
        try:
            while True:
                if cam_manager.latest_frame is None:
                    continue
                frame = cam_manager.latest_frame
                detections = detector.detect(frame)
                action_index = agent.choose_action(state)
                action = ACTIONS[action_index]

                next_state, reward, done, _ = env.step(action, detections)
                state = next_state
        except KeyboardInterrupt:
            print("\nExiting\n")
    finally:
        env.cleanup()

if __name__ == "__main__":
    main()