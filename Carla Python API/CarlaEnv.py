import numpy as np
import time
import glob
import os
import sys
import carla
import cv2
import random

class CarlaEnv:
    def __init__(self, host='localhost', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []
        self.vehicle = None
        self.camera = None

    def spawn_vehicle(self):
        vehicle_bp = self.blueprint_library.filter('model3')[0] # spawn tesla model3
        # print(vehicle_bp) # test output
        spawn_point = random.choice(self.world.get_map().get_spawn_points()) # random spawn
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(True) # just to test, remove this later
        self.actor_list.append(self.vehicle)

    def attach_camera(self, process_img):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", 640)
        camera_bp.set_attribute("image_size_y", 480)
        camera_bp.set_attribute("fov", "100")
        cam_transform = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=2.4), # 1.5m front, 2.4m up
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )
        self.camera = self.world.spawn_actor(camera_bp, cam_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(lambda data: process_img(data))
        
    def cleanup(self):
        for actor in self.actor_list:
            actor.destroy()
        print("CARLA Cleaned Up")
