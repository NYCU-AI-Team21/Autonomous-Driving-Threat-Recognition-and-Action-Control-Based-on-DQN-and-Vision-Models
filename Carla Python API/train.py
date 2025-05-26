import torch
import time
import carla
import random
import numpy as np
from collections import deque

from config import CONFIG
from CarlaEnv import CarlaEnv
from DQN import DQNAgent
from Control import CarlaControl
from YOLO import YOLODetector
from CamManager import CamManager

state_size = 4
action_size = 4

agent = DQNAgent(state_size, action_size)
detector = YOLODetector()
cam_manager = CamManager()
env = CarlaEnv()

memory = deque(maxlen=CONFIG['memory_size'])

env.attach_camera(cam_manager.process_img)

for episode in range(CONFIG['max_episode']):
    # 初始化 state 多維，依你 step() 回傳格式調整
    state = [0.0, 0, 0, -1]  
    total_reward = 0

    for step in range(CONFIG['max_steps']):
        action = agent.choose_action(state)

        # 抓 YOLO 偵測結果
        frame = cam_manager.latest_frame()
        if frame is None:
            continue
        detections = detector.detect(frame)

        # 判斷交通燈 (如果你還想在 step() 判斷，也可以改傳進 step)
        traffic_light_state = "Green"
        if env.vehicle.get_traffic_light_state() == carla.TrafficLightState.Red:
            traffic_light_state = "Red"

        next_state, reward, done, _ = env.step(action, detections, traffic_light_state)

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(memory) >= CONFIG['batch_size']:
            batch = random.sample(memory, CONFIG['batch_size'])
            for b in batch:
                agent.train_step(*b, gamma=CONFIG['gamma'])

        if done:
            print(f"Episode {episode} ended at step {step} with reward {total_reward}")
            break

    if episode % CONFIG['target_update'] == 0:
        torch.save(agent.model.state_dict(), f"model/dqn_ep{episode}.pth")

env.close()
