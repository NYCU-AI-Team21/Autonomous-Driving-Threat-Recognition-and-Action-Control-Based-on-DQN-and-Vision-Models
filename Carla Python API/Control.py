import carla

class CarlaControl:
    def __init__(self, vehicle):
        self.vehicle = vehicle

    def apply_action(self, action):
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=action[0],
            steer=action[1],
            brake=action[2],
            revers=action[3]
        ))