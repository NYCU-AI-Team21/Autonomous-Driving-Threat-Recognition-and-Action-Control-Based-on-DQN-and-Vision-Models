from flask import Flask, Response
from flask_cors import CORS
from camera_share import frame_store

app = Flask(__name__)
CORS(app)

@app.route('/image')
def get_latest_frame():
    with frame_store.frame_lock:
        if frame_store.latest_encoded_frame is None:
            # print("🔍 latest_encoded_frame is None")
            return Response("No frame yet", status=503)
        else:
            # print("📤 Sending image from camera_server.py")
            return Response(frame_store.latest_encoded_frame, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
