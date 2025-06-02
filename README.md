# Autonomous-Driving-Threat-Recognition-and-Action-Control-Based-on-DQN-and-Vision-Models
Intro to AI Team21 Final Project

This project explores how different vision model and control influence autonomous driving behavior in a simulated environment, CARLA. 
We focus on decision-making using deep reinforcement learning (DQN) and visual perception using both YOLO and CNN.

## Project Overview

Our goal is to train an agent to drive safely using image input and reinforcement learning.
The agent was evaluated based on:

- **Correct cornering rate** (how often it correctly turns)
- **Lane alignment accuracy** (how well it stayed within the lane)
- **Red light violation rate** (how often it failed to stop at red lights)

We implemented and compared three models:

1. **YOLO + DQN**
   Using YOLOv5, with a standard DQN.

3. **YOLO + DQN (with memory buffer & target network)**  
   Improved version using experience replay and a target network to stabilize learning.

4. **CNN + DQN (with memory buffer & target network)**  
   Using a CNN trained end-to-end to directly process raw image frames and output driving decisions.

## Results (100 Trials)

| Metric                 | YOLO + DQN (a) | YOLO + DQN + Memory & Target (b) | CNN + DQN + Memory & Target (c) |
|------------------------|----------------|----------------------------------|----------------------------------|
| Correct Cornering Rate | 22%            | 30%                              | 51%                              |
|  Lane Alignment        | 35%            | 41%                              | 74%                              |
|  Red Light Violations  | 45%            | 40%                              | 31%                              |

## Project Structure

project/
├── Carla Python API/ # main codes for training and evaluation
│ ├── CamManger.py
│ ├── CarlaEnv.py
│ ├── Control.py
│ ├── DModel.py
│ ├── DQN.py
│ ├── YOLO.py
│ ├── camera_server.py
│ ├── camer_share.py
│ ├── config.py
│ ├── display.py
│ ├── main.py
│ ├── manual_control.py
│ ├── plot.py
│ ├── train.py
├── YOLO_model/
│ ├── best.pt
├── backup/Carla Python API/ # Backup of earlier YOLO-only version
├── README.md # This file
├── requirements.txt # Python dependencies
