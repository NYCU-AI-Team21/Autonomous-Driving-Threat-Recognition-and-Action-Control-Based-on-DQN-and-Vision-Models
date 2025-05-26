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

    def process_img(self, image, SHOW_IMG):
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((640, 480, 4)) # into RGBA, 640 * 480 img
        img = img[:, :, :3]
        if SHOW_IMG:
            img_GBR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Carla Camera", img_GBR)
            cv2.waitKey(1)
        self.latest_frame = img # RGB ndarray
        return img
