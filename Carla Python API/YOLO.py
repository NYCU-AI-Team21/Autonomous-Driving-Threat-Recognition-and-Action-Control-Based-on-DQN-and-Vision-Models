import torch
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class YOLODetector:
    def __init__(self, model_path='../YOLO_model/best.pt'):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(device)

    def detect(self, frame):
        if frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        results = self.model(frame_rgb)
        detections = results.pandas().xyxy[0]
        return detections