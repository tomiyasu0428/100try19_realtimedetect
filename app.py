from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import base64
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 新しいObjectDetectorのインポート
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# モデル初期化
try:
    base_options = python.BaseOptions(model_asset_path="models/efficientdet_lite0.tflite")
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5, max_results=5)
    detector = vision.ObjectDetector.create_from_options(options)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("video_frame")
def handle_frame(data):
    try:
        # フレーム処理
        image_data = base64.b64decode(data.split(",")[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe用画像形式に変換
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # 検出実行
        detection_result = detector.detect(mp_image)

        # 結果処理
        processed_results = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            category = detection.categories[0]
            processed_results.append(
                {
                    "label": category.category_name,
                    "score": float(category.score),
                    "bbox": {
                        "x": bbox.origin_x,
                        "y": bbox.origin_y,
                        "width": bbox.width,
                        "height": bbox.height,
                    },
                }
            )

        logger.info(f"Detected {len(processed_results)} objects")
        socketio.emit("detection_update", processed_results)

    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        socketio.emit("error", {"message": str(e)})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True, allow_unsafe_werkzeug=True)
