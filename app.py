from flask import Flask, render_template, request, jsonify
import cv2
import base64
import numpy as np
import mediapipe as mp
from io import BytesIO
from PIL import Image

app = Flask(__name__, static_folder="static", template_folder="templates")

# MediaPipe Face Mesh setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                            max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

# Landmark indices for eye (MediaPipe FaceMesh)
# Using commonly used landmarks for EAR-like calc
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # p1..p6
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # p1..p6

def decode_base64_image(data_url):
    # data_url: "data:image/png;base64,...."
    header, encoded = data_url.split(',', 1)
    binary = base64.b64decode(encoded)
    img = Image.open(BytesIO(binary)).convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def landmark_to_point(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h])

def eye_ear(landmarks, indices, w, h):
    # indices order: p1, p2, p3, p4, p5, p6
    pts = [landmark_to_point(landmarks[i], w, h) for i in indices]
    p1, p2, p3, p4, p5, p6 = pts
    # vertical distances
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    # horizontal distance
    hdist = np.linalg.norm(p1 - p4)
    if hdist == 0:
        return 0.0
    ear = (v1 + v2) / (2.0 * hdist)
    return float(ear)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        img_data = data.get('image')
        if img_data is None:
            return jsonify({'error': 'no image provided'}), 400

        frame = decode_base64_image(img_data)
        h, w, _ = frame.shape
        # Convert to RGB for MediaPipe
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return jsonify({'found_face': False, 'left_ear': None, 'right_ear': None, 'avg_ear': None})

        landmarks = results.multi_face_landmarks[0].landmark
        left = eye_ear(landmarks, LEFT_EYE, w, h)
        right = eye_ear(landmarks, RIGHT_EYE, w, h)
        avg = (left + right) / 2.0

        # Optional: also return bounding box or landmarks count if desired
        return jsonify({
            'found_face': True,
            'left_ear': round(left, 4),
            'right_ear': round(right, 4),
            'avg_ear': round(avg, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
