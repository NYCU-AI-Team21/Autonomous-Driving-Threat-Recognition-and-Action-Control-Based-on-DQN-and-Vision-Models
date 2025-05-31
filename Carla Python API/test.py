import numpy as np
import time
import glob
import os
import sys
import carla
import cv2
import random

from config import CONFIG           # 新增的
from Control import CarlaControl    # 新增的

class CarlaEnv:
    def __init__(self, host='localhost', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []
        self.vehicle = None
        self.camera = None
        self.control = None  # 新增的
        self.collision_sensor = None # 新增的
        self.collision_happened = False  # 碰撞旗標  新增的
        for bp in self.blueprint_library.filter('vehicle.*'):
            print(bp.id)

env = CarlaEnv()