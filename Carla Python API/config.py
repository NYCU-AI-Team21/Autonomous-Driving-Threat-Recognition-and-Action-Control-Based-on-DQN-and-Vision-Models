# config.py

CONFIG = {
    'host': 'localhost',
    'port': 2000,
    'img_width': 640,
    'img_height': 480,
    'fov': 110,

    'max_episode': 500,
    'max_steps': 500,

    'gamma': 0.99,
    'lr': 1e-3,
    'batch_size': 64,
    'memory_size': 10000,
    'target_update': 10,

    'reward_collision': -100,
    'reward_forward': 1,
    'reward_brake': -0.5,

    'max_speed': 60,       # km/h
    'min_speed': 30,        # km/h
    'safe_distance': 10.0, # meters
}
