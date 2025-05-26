import torch

class YOLODetector:
    def __init__(self, model_path='/path/to/best.pt'):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    def detect(self, frame):
        results = self.model(frame)
        detections = results.pandas().xyxy[0]
        return detections
