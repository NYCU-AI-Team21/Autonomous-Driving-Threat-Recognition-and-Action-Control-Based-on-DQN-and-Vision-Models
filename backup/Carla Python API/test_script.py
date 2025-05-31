import numpy as np
import time
import glob
import os
import sys
import carla
import cv2
import random

# global variables
actor_list = []

IMAGE_H = 640
IMAGE_W = 480

def process_img(image): # process the img in ram, faster
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((IMAGE_H, IMAGE_W, 4)) # into RGBA, 640 * 480 img
    img = img[:, :, :3]
    img_GBR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Carla Camera", img_GBR)
    cv2.waitKey(1)
    return img
try:
    # init
    client = carla.Client("localhost", 2000) # connects to carla
    client.set_timeout(10.0) # server waits 10 seconds max
    world = client.get_world()
    
    # blueprint library
    blueprint_library = world.get_blueprint_library()
    
    all_spawn_points = world.get_map().get_spawn_points()
    
    # vehicle
    vehicle_bp = blueprint_library.filter('model3')[0] # spawn tesla model3
    print(vehicle_bp) # test output
    spawn_point = random.choice(all_spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True) # just to test, remove this later
    actor_list.append(vehicle)
    
    # camera attached to vehicle
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", f"{IMAGE_H}")
    camera_bp.set_attribute("image_size_y", f"{IMAGE_W}")
    camera_bp.set_attribute("fov", "100")
    cam_transform = carla.Transform(
        carla.Location(x=1.5, y=0.0, z=2.4), # 1.5m front, 2.4m up
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
    )
    camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
    actor_list.append(camera)
    camera.listen(lambda data: process_img(data))
    
    time.sleep(10)
finally:
    for actor in actor_list:
        actor.destroy()
    print("Cleaned Up.")