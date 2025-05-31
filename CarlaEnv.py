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
        #新增目的地
        self.map = self.world.get_map()
        self.destination = random.choice(self.map.generate_waypoints(2.0)).transform.location


    def spawn_vehicle(self):
        vehicle_bp = self.blueprint_library.filter('vehicle.nissan.patrol')[0]
        spawn_points = self.world.get_map().get_spawn_points()

        for attempt in range(len(spawn_points)):
            spawn_point = random.choice(spawn_points)
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                self.actor_list.append(self.vehicle)
                self.control = CarlaControl(self.vehicle, self.world)
                self._attach_collision_sensor()  # 裝上碰撞感測器
                print(f"Spawned vehicle at {spawn_point.location}")
                self.destination = random.choice(self.map.generate_waypoints(2.0)).transform.location
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
        SHOW_IMG = True # 輸出及時影像
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

    def get_front_vehicle_dist(self):
        ego_transform = self.vehicle.get_transform()
        ego_location = ego_transform.location
        ego_forward = ego_transform.get_forward_vector()

        # Get all vehicles
        vehicles = self.world.get_actors().filter('vehicle.*')

        # Set default distance
        min_distance = CONFIG["safe_distance"]
        front_vehicle = None

        for vehicle in vehicles:
            if vehicle.id == self.vehicle.id:
                continue
            target_location = vehicle.get_transform().location
            vec_to_target = target_location - ego_location
            distance = vec_to_target.length()
            
            # Check angle between forward direction and vector to target
            vec_to_target_np = np.array([vec_to_target.x, vec_to_target.y, vec_to_target.z])
            ego_forward_np = np.array([ego_forward.x, ego_forward.y, ego_forward.z])
            vec_to_target_np /= np.linalg.norm(vec_to_target_np)
            ego_forward_np /= np.linalg.norm(ego_forward_np)
            
            dot = np.dot(vec_to_target_np, ego_forward_np)
            
            # Only consider vehicles in front (within ~45 degrees cone)
            if dot > 0.7 and distance < min_distance:
                min_distance = distance
                front_vehicle = vehicle

        if front_vehicle is not None:
            print("Vehicle ahead at distance:", min_distance)
            return min_distance
        else:
            return None

    def get_ahead_traffic_light(self, max_dist=40.0, max_vehicle_angle=33.3, max_facing_angle=120.0):
        """
        Return the traffic light that is:
        - within `max_dist` meters,
        - within `max_vehicle_angle` of the vehicle's forward direction,
        - *and* whose forward vector is pointing toward the vehicle within `max_facing_angle`.
        """
        vehi_loc   = self.vehicle.get_location()
        vehi_fwd   = self.vehicle.get_transform().get_forward_vector()
        vehi_fwd_np = np.array([vehi_fwd.x, vehi_fwd.y, vehi_fwd.z])
        vehi_fwd_np /= np.linalg.norm(vehi_fwd_np)

        best_tl    = None
        best_dist  = float('inf')

        for tl in self.world.get_actors().filter('traffic.traffic_light*'):
            tl_loc  = tl.get_transform().location
            vec     = np.array([tl_loc.x - vehi_loc.x,
                                tl_loc.y - vehi_loc.y,
                                tl_loc.z - vehi_loc.z])
            dist    = np.linalg.norm(vec)
            if dist > max_dist:
                continue

            vec_norm = vec / dist

            # 1) Is the light roughly in front of the car?
            vehicle_angle = np.degrees(
                np.arccos(np.clip(np.dot(vehi_fwd_np, vec_norm), -1.0, 1.0))
            )
            # print("In front of the car? ", vehicle_angle)
            if vehicle_angle > max_vehicle_angle:
                continue

            # 2) Is the light's forward vector pointing *toward* the vehicle?
            light_fwd = tl.get_transform().get_forward_vector()
            light_fwd_np = np.array([light_fwd.x, light_fwd.y, light_fwd.z])
            light_fwd_np /= np.linalg.norm(light_fwd_np)

            # The light faces INTO the intersection, so toward the car is opposite its fwd:
            facing_dot = np.dot(-light_fwd_np, vec_norm)
            facing_angle = np.degrees(np.arccos(np.clip(facing_dot, -1.0, 1.0)))
            # print("Facing the car? ", facing_angle)
            if facing_angle > max_facing_angle:
                continue

            if dist < best_dist:
                best_dist = dist
                best_tl   = tl

        return best_tl, best_dist

    def step(self, action, detections):
        vehicle_location = self.vehicle.get_location()
        before_dis = vehicle_location.distance(self.destination)
        for _ in range(3):
            self.control.apply_action(action)
            self.world.tick()
        
        speed = self.get_speed()

        # 初始 reward 和 done
        reward = 0.0
        done = False

        if self.collision_happened:
            reward -= 20.0
            done = True
            self.collision_happened = False  # 碰撞判定一次後清除，避免持續判定
        
        # Traffic lights
        TL_MAX_DIST  = 25.0
        traffic_light, dist = self.get_ahead_traffic_light()
        # traffic_light = self.vehicle.get_traffic_light()
        red_light_flag = 0
        if traffic_light:
            tl_state = traffic_light.get_state()
            print("Traffic light detected")
            print("Traffic light state : ", tl_state)

            tl_loc = traffic_light.get_transform().location
            dist_to_tl = abs(self.vehicle.get_location().distance(tl_loc) - 15.0)

            print(f"Distance to imaginary stop line: {dist_to_tl:.2f} m")
            if dist_to_tl < TL_MAX_DIST and tl_state == carla.TrafficLightState.Red:
                red_light_flag = 1
                proximity_factor = max(0.0, (TL_MAX_DIST - dist_to_tl) / TL_MAX_DIST)

                if speed > 5:
                    # penalty scales with speed and how close you are
                    penalty = 150.0 * proximity_factor * (speed / 30)
                    reward -= penalty
                    print(f"Ran red light → speed={speed:.2f}, dist={dist_to_tl:.2f}, penalty={penalty:.1f}")
                    # you can choose to kill the episode only if very close
                    if dist_to_tl < 10.0:  
                        done = True

                else:
                    reward += 30.0
                    print("Stopped at red light, good!")
        else:
            # No traffic light active
            if CONFIG["min_speed"] <= speed <= CONFIG["max_speed"]:
                # print("Good speed")
                reward += 0.8
            elif speed > CONFIG["max_speed"]:
                # print("Too fast!")
                reward -= 4.0
            else:
                # print("Too slow!")
                reward -= 2.0

        
        off_road = None

        waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=False)

        if waypoint is not None and waypoint.lane_type == carla.LaneType.Driving:
            reward += 1.5  #在正規車道
            off_road = True
        else:
            # print("Off road!")
            reward -= 10.0  #在非非正規車道
            off_road = False

        if waypoint is not None:
            def unit(v):
                norm = np.linalg.norm(v)
                return v / norm if norm > 0 else v

            # lane direction
            lane_forward = waypoint.transform.get_forward_vector()
            lane_np = np.array([lane_forward.x, lane_forward.y])
            lane_unit = unit(lane_np)

            # vehicle motion direction
            vehicle_v = self.vehicle.get_velocity()
            vehi_v_np = np.array([vehicle_v.x, vehicle_v.y])
            speed = np.linalg.norm(vehi_v_np)

            if speed < 0.1:
                angle_signed = None
            else:
                vehi_unit = vehi_v_np / speed
                # dot = cos(theta)
                dot = np.clip(np.dot(vehi_unit, lane_unit), -1.0, 1.0)
                # det = cross_z = lane_x * veh_y - lane_y * veh_x
                det =   lane_unit[0] * vehi_unit[1] - lane_unit[1] * vehi_unit[0]
                # signed angle in radians
                theta = np.arctan2(det, dot)
                # convert to degrees, and flip sign if you want positive = “too right”
                angle_signed = -np.degrees(theta)

                # print(f"Signed angle vs lane: {angle_signed:.2f}°")

            # now use angle_signed for reward
            if angle_signed is not None:
                if abs(angle_signed) < 2:
                    reward += 0.1
                elif abs(angle_signed) < 10:        # positive → veering to the right
                    reward -= 5.0 * (abs(angle_signed) / 45)  # scale penalty
                else:                         # negative → veering to the left
                    reward -= 10.0 * (abs(angle_signed) / 45)
   

        vehicle_count = 0
        front_distance = self.get_front_vehicle_dist()

        if front_distance is not None and front_distance < CONFIG["safe_distance"]:
            print("Not in safe distance!")
            reward -= 0.5

        # for _, box in detections.iterrows():
        #     class_name = box['name']
        #     y_bottom = box['ymax']  # DataFrame 用 ymax 代表框的底部y座標
        #     x_center = (box['xmin'] + box['xmax']) / 2

        #     if class_name in ['car', 'truck', 'bus', 'pedestrian']:
        #         vehicle_count += 1
        #         if 240 < x_center < 400:
        #             dist = 480 - y_bottom
        #             if (front_distance is None) or (dist < front_distance):
        #                 front_distance = dist
        #     if front_distance is not None:
        #         if front_distance < CONFIG["safe_distance"]:  
        #             print("Not in safe distance!")
        #             reward -= 0.5

        #計算目的地距離
        '''
        dest_distance = vehicle_location.distance(self.destination)
        if dest_distance< before_dis:
            reward += 1
        else:
            reward -=1
        #如果接近目的地
        if dest_distance < CONFIG["dest_arrival_threshold"]:  # 你可以在 config 裡定義 5.0 (單位公尺)
            reward += 30.0
            done = True
        '''

        # 多維 state 回傳
        next_state = [
            speed,
            vehicle_count,
            1 if off_road else 0,
            red_light_flag,
            front_distance if front_distance is not None else 100.0
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


