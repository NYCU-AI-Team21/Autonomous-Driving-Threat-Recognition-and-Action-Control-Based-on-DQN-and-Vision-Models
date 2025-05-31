import torch
import cv2
import sys
import os
import pathlib

# 自動加入 yolov5-master 路徑
yolov5_repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolov5-master'))
sys.path.append(yolov5_repo_path)

# ✅ Monkey patch to support Windows-trained models on Linux
pathlib.WindowsPath = pathlib.PosixPath

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class YOLODetector:
    def __init__(self, model_path='../YOLO_model/best.pt'):
        # 使用 torch.hub 載入本地 yolov5
        self.model = torch.hub.load(yolov5_repo_path, 'custom', path=model_path, source='local')
        self.model.to(device).eval()

    def detect(self, frame):
        # 檢查是否需要轉 RGB
        if frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        with torch.no_grad():
            results = self.model(frame_rgb)

        detections = results.pandas().xyxy[0]
        return detections
