import threading

class FrameStore:
    def __init__(self):
        self.latest_encoded_frame = None
        self.frame_lock = threading.Lock()

frame_store = FrameStore()
