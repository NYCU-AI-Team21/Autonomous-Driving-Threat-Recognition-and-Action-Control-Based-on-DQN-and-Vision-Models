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

    def spawn_vehicle(self):
        vehicle_bp = self.blueprint_library.filter('vehicle.nissan.patrol')[0]
        spawn_points = self.world.get_map().get_spawn_points()

        for attempt in range(len(spawn_points)):
            spawn_point = random.choice(spawn_points)
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                self.actor_list.append(self.vehicle)
                self.control = CarlaControl(self.vehicle)
                self._attach_collision_sensor()  # 裝上碰撞感測器
                print(f"Spawned vehicle at {spawn_point.location}")
                return  # spawn 成功就離開函式
            except RuntimeError:
                print(f"Spawn failed at {spawn_point.location}, retrying...")

        # 如果嘗試所有點都失敗，拋錯
        raise RuntimeError("Spawn vehicle failed: no free spawn points available.")
#--------------------------------------------------------------------------碰撞偵測(不確定行不行) begin
    def _attach_collision_sensor(self):
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.actor_list.append(self.collision_sensor)
        # 感測器事件回呼
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _on_collision(self, event):
        self.collision_happened = True
#--------------------------------------------------------------------------碰撞偵測(不確定行不行)  end
    def attach_camera(self, process_img):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "480")
        camera_bp.set_attribute("fov", "100")
        camera_bp.set_attribute("sensor_tick", str(1.0 / 10)) # set to 10 fps for now
        cam_transform = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=2.4),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )
        SHOW_IMG = False # 輸出及時影像
        self.camera = self.world.spawn_actor(camera_bp, cam_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(lambda data: process_img(data, SHOW_IMG))
#------------------------------------------------------------------------------------新增的 begin
    def get_speed(self):
        """取得目前車速，單位 km/h"""
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        return speed

    def reset(self):
        self.cleanup()
        time.sleep(1.0)
        self.spawn_vehicle()

    def step(self, action, detections, traffic_light_state):
        self.control.apply_action(action)
        time.sleep(0.1)

        speed = self.get_speed()

        # 初始 reward 和 done
        reward = 0.0
        done = False

        if self.collision_happened:
            reward -= 2.0
            done = True
            self.collision_happened = False  # 碰撞判定一次後清除，避免持續判定

        # 速度獎懲
        if CONFIG["min_speed"] <= speed <= CONFIG["max_speed"]:
            reward += 0.5
        elif speed > 50:
            reward -= 0.5
        else:
            reward -= 0.5

        # 交通燈判定
        traffic_light_state = self.vehicle.get_traffic_light_state()
        if traffic_light_state == carla.TrafficLightState.Red:
            reward -= 1.0
        #    done = True

        vehicle_count = 0
        off_road = False
        front_distance = None

        for _, box in detections.iterrows():
            class_name = box['name']
            y_bottom = box['ymax']  # DataFrame 用 ymax 代表框的底部y座標
            x_center = (box['xmin'] + box['xmax']) / 2

            if class_name in ['car', 'truck', 'bus', 'pedestrian']:
                vehicle_count += 1
                if 240 < x_center < 400:
                    dist = 480 - y_bottom
                    if (front_distance is None) or (dist < front_distance):
                        front_distance = dist
            elif class_name == 'offroad':
                off_road = True
                if front_distance is not None:
                    if front_distance < 50:  
                        reward -= 0.5


        # 多維 state 回傳
        next_state = [
            speed,
            vehicle_count,
            1 if off_road else 0,
            front_distance if front_distance is not None else -1
        ]

        return next_state, reward, done, {}
    
    
    def close(self):
        self.cleanup()
    
#------------------------------------------------------------------------------------新增的 end

    def cleanup(self):
        for actor in self.actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actor_list = []
        self.vehicle = None
        self.camera = None
        self.control = None
        self.collision_sensor = None
        self.collision_happened = False
        print("CARLA Cleaned Up")


