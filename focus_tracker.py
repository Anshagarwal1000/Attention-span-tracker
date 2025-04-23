from flask import Flask, render_template, request
import threading
import time
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist

app = Flask(__name__)

# Global control flag
stop_tracking = False

# Landmark indices
LEFT_IRIS_IDX = [468, 469, 470, 471]
RIGHT_IRIS_IDX = [473, 474, 475, 476]
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
LEFT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_TOP_BOTTOM = [386, 374]
EAR_THRESHOLD = 0.25

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_center(landmarks, indices, w, h):
    xs = [landmarks[i].x * w for i in indices]
    ys = [landmarks[i].y * h for i in indices]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

def is_gaze_center(pupil_x, eye_left_x, eye_right_x, pupil_y, eye_top_y, eye_bottom_y):
    eye_width = abs(eye_right_x - eye_left_x)
    eye_height = abs(eye_bottom_y - eye_top_y)
    center_x = (eye_left_x + eye_right_x) / 2
    center_y = (eye_top_y + eye_bottom_y) / 2
    return (abs(pupil_x - center_x) < (eye_width * 0.25) and
            abs(pupil_y - center_y) < (eye_height * 0.25))

def track_focus():
    global stop_tracking
    stop_tracking = False

    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                       refine_landmarks=True,
                                       min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5)

    focus_start_time = None
    total_focus_time = 0
    focused = False
    timestamps = []
    attentiveness = []
    session_start_time = time.time()

    while not stop_tracking:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time() - session_start_time
        timestamps.append(current_time)
        is_focused = False

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lm = face_landmarks.landmark

                left_eye = [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_EYE_IDX]
                right_eye = [(int(lm[i].x * w), int(lm[i].y * h)) for i in RIGHT_EYE_IDX]

                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear > EAR_THRESHOLD:
                    left_pupil = get_center(lm, LEFT_IRIS_IDX, w, h)
                    right_pupil = get_center(lm, RIGHT_IRIS_IDX, w, h)

                    left_corners = [int(lm[i].x * w) for i in LEFT_EYE_CORNERS]
                    right_corners = [int(lm[i].x * w) for i in RIGHT_EYE_CORNERS]

                    left_tb = [int(lm[i].y * h) for i in LEFT_EYE_TOP_BOTTOM]
                    right_tb = [int(lm[i].y * h) for i in RIGHT_EYE_TOP_BOTTOM]

                    left_centered = is_gaze_center(left_pupil[0], *left_corners,
                                                   left_pupil[1], *left_tb)
                    right_centered = is_gaze_center(right_pupil[0], *right_corners,
                                                    right_pupil[1], *right_tb)

                    is_focused = left_centered and right_centered

                    cv2.circle(frame, left_pupil, 2, (0, 255, 0), -1)
                    cv2.circle(frame, right_pupil, 2, (0, 255, 0), -1)

        if is_focused:
            if not focused:
                focus_start_time = time.time()
                focused = True
            attentiveness.append(1)
        else:
            if focused:
                total_focus_time += time.time() - focus_start_time
                focused = False
            attentiveness.append(0)

        focus_display = total_focus_time
        if focused and focus_start_time:
            focus_display += time.time() - focus_start_time

        cv2.putText(frame, f'Focus Time: {focus_display:.2f}s', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Focus Analyzer (Gaze All Directions)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if focused and focus_start_time:
        total_focus_time += time.time() - focus_start_time

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    print(f'Total Focus Time: {total_focus_time:.2f} seconds')

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, attentiveness, color='dodgerblue', linewidth=2)
    plt.fill_between(timestamps, attentiveness, color='lightgreen', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Attentiveness')
    plt.title('Focus Analysis Over Time')
    plt.yticks([0, 1], ['Distracted', 'Focused'])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    threading.Thread(target=track_focus).start()
    return "Focus tracking started"

@app.route('/stop', methods=['POST'])
def stop():
    global stop_tracking
    stop_tracking = True
    return "Focus tracking stopped"

if __name__ == '__main__':
    app.run(debug=True)
