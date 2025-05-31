import carla
import numpy as np

def get_speed(vehicle):
        """取得目前車速，單位 km/h"""
        velocity = vehicle.get_velocity()
        speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        return speed

class CarlaControl:
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world

    def apply_action(self, action):
        throttle = action[0]
        steer    = action[1]
        brake    = action[2]
        reverse  = action[3]

        #避免轉彎沒速度
        if steer != 0 and get_speed(self.vehicle) < 5:
             throttle = 0.5
    
        if get_speed(self.vehicle) > 40:
            throttle = 0.0

        # 保證 throttle 和 brake 不同時作用
        if throttle > 0.0:
            brake = 0.0

        # 如果 reverse 狀態切換，先停車再切
        '''
        current_reverse = self.vehicle.get_control().reverse
        if reverse != current_reverse:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, reverse=current_reverse))
            self.world.tick()  # 確保車子停穩
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, reverse=reverse))
            self.world.tick()  # 確保 reverse 狀態改好    


        '''

        if brake == 0.0:
            target_speed = 25
            speed = get_speed(self.vehicle)
            speed_error = target_speed - speed                
            throttle = np.clip(speed_error * 0.5, 0.0, 1.0)  # 如果速度小於目標，給油門
            if speed > 25:
                throttle = 0.0
                brake = np.clip((speed - target_speed) * 0.5, 0.0, 1.0)  # 超速就踩煞車
        # else:
        #    print("Braking")
        
        # 最後套用 action
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            reverse=reverse
        ))