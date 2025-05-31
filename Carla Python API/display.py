# client_display.py
import requests
import cv2
import numpy as np
import time

while True:
    try:
        response = requests.get("http://localhost:5000/image", timeout=1)
        if response.status_code == 200:
            jpg_data = response.content
            np_array = np.frombuffer(jpg_data, np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            cv2.imshow('Remote Camera', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            print("Waiting for frame...")
    except Exception as e:
        print(f"Error fetching frame: {e}")
        time.sleep(1)

cv2.destroyAllWindows()
