from flask import Flask, render_template, Response, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, JWTManager
import cv2
from deepface import DeepFace
import numpy as np
import os
import datetime
import time

app = Flask(__name__)

# Cấu hình JWT
app.config['JWT_SECRET_KEY'] = 'your_secret_key'
jwt = JWTManager(app)

# Global variables
video_capture = None
emotion_data = []
frame_count = 0
process_interval = 10
frame_delay = 0.1  # Giảm tốc độ xử lý video

# Initialize camera
def initialize_camera():
    global video_capture
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        video_capture = None

initialize_camera()

# Analyze frame function
def analyze_frame(frame):
    global frame_count, emotion_data
    frame_count += 1
    
    if frame_count % process_interval != 0:
        return frame
    
    try:
        results = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            detector_backend="opencv",
            enforce_detection=False
        )

        for face in results:
            region = face.get('region', {})
            if not region:
                continue

            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            dominant_emotion = face.get('dominant_emotion', "unknown")
            emotion_score = face.get('emotion', {}).get(dominant_emotion, 0)

            if emotion_score > 0.9 and dominant_emotion != "neutral":
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"static/saved_frames/{dominant_emotion}_{timestamp}.jpg"
                if not os.path.exists("static/saved_frames"):
                    os.makedirs("static/saved_frames")
                cv2.imwrite(filename, frame)

                emotion_data.append({
                    "timestamp": timestamp,
                    "emotion": dominant_emotion,
                    "score": emotion_score,
                    "image_path": filename
                })

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{dominant_emotion}: {emotion_score:.2f}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error: {str(e)}")

    return frame

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
                    
                    time.sleep(frame_delay)  # Thêm độ trễ để làm chậm video
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
