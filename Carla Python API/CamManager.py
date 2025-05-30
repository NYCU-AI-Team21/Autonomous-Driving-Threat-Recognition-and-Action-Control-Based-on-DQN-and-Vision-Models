import numpy as np
import time
import glob
import os
import sys
import carla
import cv2
import random
from camera_share import frame_store

class CamManager:
    def __init__(self):
        self.latest_frame = None

    def process_img(self, image, SHOW_IMG=True):
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((image.height, image.width, 4))[:, :, :3]

        if SHOW_IMG:
            with frame_store.frame_lock:
                frame_store.latest_encoded_frame = cv2.imencode('.jpg', img)[1].tobytes()
            cv2.imshow("CARLA Camera", img)
            cv2.waitKey(1)
        
        resized_img = cv2.resize(img, (84, 84))
        self.latest_frame = resized_img
        return resized_img
        
