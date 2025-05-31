import torch
import time
from tqdm import tqdm
import carla
import random
import numpy as np
from collections import deque
import cv2
import matplotlib.pyplot as plt
import os

import plot
from config import CONFIG
from CarlaEnv import CarlaEnv
from DQN import DQNAgent
from Control import CarlaControl
from YOLO import YOLODetector
from CamManager import CamManager

import logging
import threading
from camera_server import app

# 關閉 werkzeug HTTP 請求日誌
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# 啟動 Flask server thread，不輸出一般請求日誌
threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True).start()

state_size = 5
action_size = 7

ACTIONS = [
    (0.25, 0.0, 0.0, False),  # 直行加速
    (0.0, 0.0, 0.2, False),  # 直行減速
    (0.0, -0.6, 0.03, False),  # 左轉大
    (0.0, -0.4, 0.05, False),  # 左轉小
    (0.0, 0.6, 0.03, False),  # 右轉大
    (0.0, 0.4, 0.05, False),  # 右轉小
    (0.1, 0.0, 0.0, True),   # 倒車
]

agent = DQNAgent(state_size, action_size, CONFIG['epsilon'], CONFIG['epsilon_min'], CONFIG['epsilon_decay'], CONFIG['target_update_freq'])

detector = YOLODetector()
env = CarlaEnv()

memory = deque(maxlen=CONFIG['memory_size'])
episode_rewards = []
episode_steps = []
losses = [] 
best_reward = -float('inf')

try:
    for episode in tqdm(range(CONFIG['max_episode'])):
        total_reward = 0
        cam_manager = CamManager()
        env.cleanup()
        env.spawn_vehicle()
        env.attach_camera(cam_manager.process_img)

        while cam_manager.latest_frame is None:
            time.sleep(0.05)

        state = cam_manager.latest_frame

        for step in range(1, CONFIG['max_steps'] + 1):
            action_index = agent.choose_action(state)
            action = ACTIONS[action_index]
            # print(f"step = {step}, action_index = {action_index}")

            frame = cam_manager.latest_frame
            if frame is None:
                print(f"Episode {episode} step {step}: no frame yet")
                time.sleep(0.01)
                continue

            detections = detector.detect(frame)
            no_use, reward, done, _ = env.step(action, detections)
            next_state = cam_manager.latest_frame

            memory.append((state, action_index, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) >= CONFIG['batch_size']:
                batch = random.sample(memory, CONFIG['batch_size'])
                for b in batch:
                    loss = agent.train_step(*b, gamma=CONFIG['gamma'])
                    losses.append(loss)

            if done:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(step+1)
        print(f"Episode {episode} ended at step {step} with reward {total_reward}", flush=True)

        if episode % CONFIG['target_update'] == 0:
            torch.save(agent.model.state_dict(), f"./model/dqn_ep{episode}.pth")
            print(f"Model saved at {episode}")

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.model.state_dict(), f"./model/best_model.pth")
            print(f"New best model saved with reward {total_reward}")

except KeyboardInterrupt:
    print("\n[INFO] KeyboardInterrupt detected. Cleaning up...")
    env.cleanup()
    env.close()
    cv2.destroyAllWindows()
    print("[INFO] Environment closed and resources released.")

finally:
    print("[INFO] Saving final training curves...")
    plot.save_training_curves(episode_rewards, episode_steps, losses)
    print("[INFO] Training terminated cleanly.")