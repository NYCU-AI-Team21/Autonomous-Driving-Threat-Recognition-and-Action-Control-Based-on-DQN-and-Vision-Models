import numpy as np
import time
import glob
import os
import sys
import carla
import cv2
import random

class CamManager:
    def __init__(self):
        self.latest_frame = None

    def process_img(self, image, SHOW_IMG=False):
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((480, 640, 4))  # 注意這裡
        img = img[:, :, :3]  # RGB only
        if SHOW_IMG:
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Carla Camera", img_BGR)
            cv2.waitKey(1)
        self.latest_frame = img
        return img
