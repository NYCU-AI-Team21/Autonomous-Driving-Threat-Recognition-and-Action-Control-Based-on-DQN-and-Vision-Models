import numpy as np
import time
import glob
import os
import sys
import carla
import cv2
import random

from Control import CarlaControl
from YOLO import YOLODetector
from CarlaEnv import CarlaEnv
from CamManager import CamManager
from DQN import DQNAgent

def main():
    env = CarlaEnv()
    cam_manager = CamManager()
    detector = YOLODetector()
    agent = DQNAgent()
    
    try:
        env.spawn_vehicle()
        env.attach_camera(cam_manager.process_img)
        controller = CarlaControl(env.vehicle)

        while True:
            if cam_manager.latest_frame is None:
                continue
            # frame = cam_manager.latest_frame
            # detections = detector.detect(frame)
            # state = get state from YOLO
            # action = DQN choose action
            # controller.apply_action(action)
    finally:
        env.cleanup()

if __name__ == "__main__":
    main()