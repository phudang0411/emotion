from flask import Flask, render_template, Response, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, JWTManager
import cv2
from deepface import DeepFace
import joblib
import numpy as np
import threading
import os
import datetime

app = Flask(__name__)

# Cấu hình JWT
app.config['JWT_SECRET_KEY'] = 'your_secret_key'  # Đổi secret key thành key riêng của bạn
jwt = JWTManager(app)

# Load model và label encoder
clf = joblib.load("classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Global variables
video_capture = None
frame_output = None
emotion_data = []
frame_count = 0
process_interval = 10
unique_identities = {}

# Initialize camera
def initialize_camera():
    global video_capture
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        video_capture = None

initialize_camera()

# Lưu trạng thái các bounding box trong khung hình trước
last_faces = []
frame_life = 5

# Analyze frame function
def analyze_frame(frame):
    global frame_count, emotion_data, last_faces, unique_identities
    frame_count += 1

    if frame_count % process_interval != 0:
        for face in last_faces:
            x1, y1, x2, y2, identity, emotion, emotion_score = face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{identity} ({emotion}: {emotion_score:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    try:
        results = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            detector_backend="opencv",
            enforce_detection=False
        )

        current_faces = []
        for face in results:
            region = face.get('region', {})
            if not region:
                continue

            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            dominant_emotion = face.get('dominant_emotion', "unknown")
            emotion_score = face.get('emotion', {}).get(dominant_emotion, 0)

            y1, y2 = max(0, y), min(frame.shape[0], y + h)
            x1, x2 = max(0, x), min(frame.shape[1], x + w)
            cropped_face = frame[y1:y2, x1:x2]

            if cropped_face.size == 0:
                continue

            embedding_result = DeepFace.represent(
                img_path=np.array(cropped_face),
                model_name="VGG-Face",
                enforce_detection=False
            )
            feature_vector = np.array(embedding_result[0]['embedding']).reshape(1, -1)

            probas = clf.predict_proba(feature_vector)[0]
            predicted_label = np.argmax(probas)
            identity = label_encoder.inverse_transform([predicted_label])[0]

            if probas[predicted_label] < 0.35:
                identity = "unknown"

            if identity != "unknown":
                if identity not in unique_identities:
                    unique_identities[identity] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            current_faces.append((x1, y1, x2, y2, identity, dominant_emotion, emotion_score))

            if emotion_score > 0.9 and dominant_emotion != "neutral":
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"static/saved_frames/{identity}_{dominant_emotion}_{timestamp}.jpg"
                if not os.path.exists("static/saved_frames"):
                    os.makedirs("static/saved_frames")
                cv2.imwrite(filename, frame)

                emotion_data.append({
                    "timestamp": timestamp,
                    "identity": identity,
                    "emotion": dominant_emotion,
                    "score": emotion_score,
                    "image_path": filename
                })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{identity} ({dominant_emotion}: {emotion_score:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        last_faces = current_faces

    except Exception as e:
        print(f"Error: {str(e)}")

    if not last_faces:
        return frame

    return frame
@app.route('/')
def index():
    camera_available = video_capture is not None
    return render_template('index.html', camera_available=camera_available)
# API: Đăng nhập
@app.route('/login', methods=['POST'])
def login():
    username = request.json.get("username", "")
    password = request.json.get("password", "")

    if username == "giangvien" and password == "123456":
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({"msg": "Bad username or password"}), 401
    

# API: Xem bảng điểm danh
@app.route('/api/attendance', methods=['GET'])
@jwt_required()
def get_attendance():
    filtered_identities = {k: v for k, v in unique_identities.items() if k != "unknown"}
    return jsonify(filtered_identities), 200

@app.route('/api/emotion_frames', methods=['GET'])
@jwt_required()
def get_emotion_frames():
    try:
        # Kiểm tra nếu emotion_data không tồn tại hoặc rỗng
        if not emotion_data:
            return jsonify({"message": "No emotion frames available"}), 404

        # Duyệt qua emotion_data và chuyển đổi giá trị float32 thành float
        processed_data = []
        for data in emotion_data:
            if isinstance(data.get('score'), np.float32):
                data['score'] = float(data['score'])  # Chuyển float32 thành float
            processed_data.append(data)

        # Lọc dữ liệu: identity khác "unknown" và score > 80
        filtered_data = [
            data for data in processed_data 
            if data['identity'] != "unknown" and data['score'] > 80
        ]

        # Trả về dữ liệu đã lọc
        return jsonify(filtered_data), 200
    except Exception as e:
        # Ghi log lỗi (nếu có)
        app.logger.error(f"Error fetching emotion frames: {str(e)}")
        return jsonify({"error": "An error occurred while fetching emotion frames"}), 500


# API: Nhận thông báo khi có frame cảm xúc cao
@app.route('/api/notify_high_emotion', methods=['GET'])
@jwt_required()
def notify_high_emotion():
    try:
        # Kiểm tra nếu emotion_data không tồn tại hoặc rỗng
        if not emotion_data:
            return jsonify({"msg": "No data available to check emotions."}), 404

        # Lọc các khung hình có cảm xúc cao
        high_emotion_frames = [
            data for data in emotion_data 
            if data['score'] > 80 and data['emotion'] != "neutral"  # Ngưỡng score > 80
        ]
        
        if high_emotion_frames:
            # Chuyển đổi score từ np.float32 thành float
            for frame in high_emotion_frames:
                frame['score'] = float(frame['score'])  # Chuyển đổi kiểu score

            # Ghi log danh sách các khung hình cảm xúc cao
            app.logger.info(f"High emotion frames: {high_emotion_frames}")

            # Trả về thông báo
            return jsonify({
                "msg": "High emotion detected, notification sent.",
                "details": high_emotion_frames
            }), 200

        # Trường hợp không phát hiện cảm xúc cao
        return jsonify({"msg": "No high emotion detected."}), 200

    except Exception as e:
        # Ghi log lỗi (nếu có)
        app.logger.error(f"Error checking high emotion: {str(e)}")
        return jsonify({"error": "An error occurred while checking emotions."}), 500


@app.route('/video_feed')
def video_feed():
    if video_capture is None:
        return Response("No camera available", status=404)

    def generate():
        global video_capture
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame = analyze_frame(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/dashboard')
def dashboard():
    filtered_emotion_data = [data for data in emotion_data if data['identity'] != "unknown"]
    return render_template('dashboard.html', 
                           emotion_data=filtered_emotion_data, 
                           unique_identities=unique_identities)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['video']
    if file:
        upload_folder = "uploaded_videos"
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        video_path = os.path.join(upload_folder, file.filename)
        file.save(video_path)

        def process_video(video_path):
            cap = cv2.VideoCapture(video_path)

            def generate():
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed_frame = analyze_frame(frame)
                    _, buffer = cv2.imencode('.jpg', processed_frame)
                    frame_data = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

                cap.release()

            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

        return process_video(video_path)
    else:
        return "No file uploaded."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
