import torch
import time
from tqdm import tqdm
import carla
import random
import numpy as np
from collections import deque
import cv2

from config import CONFIG
from CarlaEnv import CarlaEnv
from DQN import DQNAgent
from Control import CarlaControl
from YOLO import YOLODetector
from CamManager import CamManager

state_size = 3
action_size = 4

ACTIONS = [
        [0.5, 0.0, 0.0],   # go straight
        [0.5, -0.3, 0.0],  # turn left
        [0.5, 0.3, 0.0],   # turn right
        [0.0, 0.0, 1.0]    # brake
    ]

agent = DQNAgent(state_size, action_size)
detector = YOLODetector()
env = CarlaEnv()

memory = deque(maxlen=CONFIG['memory_size'])

for episode in tqdm(range(CONFIG['max_episode'])):
    # 初始化 state 多維，依你 step() 回傳格式調整
    state = [0.0, 0, -1]  
    total_reward = 0
    cam_manager = CamManager() # added
    env.cleanup()
    env.spawn_vehicle()
    env.attach_camera(cam_manager.process_img)
    # print("1")
    while cam_manager.latest_frame is None:
        time.sleep(0.05)
    # print("7")
    for step in range(CONFIG['max_steps']):
        action_index = agent.choose_action(state)
        # print("6")
        action = ACTIONS[action_index]
        # print("5")
        # 抓 YOLO 偵測結果
        frame = cam_manager.latest_frame
        if frame is None:
            print(f"Episode {episode} step {step}: no frame yet")
            time.sleep(0.01)
            continue
        # print("8")
        detections = detector.detect(frame)
        # print("4")

        next_state, reward, done, _ = env.step(action, detections)

        memory.append((state, action_index, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(memory) >= CONFIG['batch_size']:
            batch = random.sample(memory, CONFIG['batch_size'])
            for b in batch:
                agent.train_step(*b, gamma=CONFIG['gamma'])

        if done:
            # print("2")
            break
    # print("3")
    print(f"Episode {episode} ended at step {step} with reward {total_reward}", flush=True)
    if episode % CONFIG['target_update'] == 0:
        torch.save(agent.model.state_dict(), f"./model/dqn_ep{episode}.pth")
    #cv2.destroyAllWindows()

env.close()
