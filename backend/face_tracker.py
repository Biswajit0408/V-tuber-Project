import cv2
import mediapipe as mp
import math
import time
import numpy as np

# =========================
# Utility functions
# =========================
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))

def normalize(val, min_val, max_val):
    return clamp((val - min_val) / (max_val - min_val), 0.0, 1.0)

def load_png(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

# =========================
# Middle finger detection
# =========================
def is_middle_finger(hand_landmarks, h):
    middle_tip = int(hand_landmarks.landmark[12].y * h)
    middle_knuckle = int(hand_landmarks.landmark[10].y * h)

    index_tip = int(hand_landmarks.landmark[8].y * h)
    index_knuckle = int(hand_landmarks.landmark[6].y * h)

    ring_tip = int(hand_landmarks.landmark[16].y * h)
    ring_knuckle = int(hand_landmarks.landmark[14].y * h)

    pinky_tip = int(hand_landmarks.landmark[20].y * h)
    pinky_knuckle = int(hand_landmarks.landmark[18].y * h)

    return (
        middle_tip < middle_knuckle and
        index_tip > index_knuckle and
        ring_tip > ring_knuckle and
        pinky_tip > pinky_knuckle
    )

# =========================
# Load avatar images
# =========================
images = {
    "neutral": load_png("avatar/neutral.png"),
    "eyes_closed": load_png("avatar/eyes_closed.png"),
    "laugh": load_png("avatar/laugh.png"),
    "monster": load_png("avatar/monster.png"),
    "middle_finger": load_png("avatar/middle_finger.png"),
}

# =========================
# MediaPipe setup
# =========================
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
WINDOW_NAME = "VTuber Avatar"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# =========================
# Cooldown system
# =========================
COOLDOWN_SECONDS = 1.5
last_action_time = 0
current_state = "neutral"

# =========================
# Main loop
# =========================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_result = face_mesh.process(rgb)
        hand_result = hands.process(rgb)

        eye_val = 1.0
        mouth_val = 0.0
        hand_raised = False
        middle_finger = False

        # =========================
        # Face processing
        # =========================
        if face_result.multi_face_landmarks:
            lm = face_result.multi_face_landmarks[0].landmark
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]

            left_eye = distance(pts[159], pts[145])
            right_eye = distance(pts[386], pts[374])
            eye_val = normalize((left_eye + right_eye) / 2, 4, 10)

            mouth = distance(pts[13], pts[14])
            mouth_val = normalize(mouth, 2, 15)

            eye_y = int(lm[159].y * h)

            # =========================
            # Hand processing
            # =========================
            if hand_result.multi_hand_landmarks:
                for hand in hand_result.multi_hand_landmarks:
                    if is_middle_finger(hand, h):
                        middle_finger = True
                        break

                    for p in hand.landmark:
                        if int(p.y * h) < eye_y:
                            hand_raised = True

        # =========================
        # State logic (PRIORITY + COOLDOWN)
        # =========================
        now = time.time()

        if middle_finger and (now - last_action_time) > COOLDOWN_SECONDS:
            current_state = "middle_finger"
            last_action_time = now

        elif hand_raised and (now - last_action_time) > COOLDOWN_SECONDS:
            current_state = "monster"
            last_action_time = now

        elif eye_val < 0.25:
            current_state = "eyes_closed"

        elif mouth_val > 0.6:
            current_state = "laugh"

        elif (now - last_action_time) > COOLDOWN_SECONDS:
            current_state = "neutral"

        # =========================
        # Render avatar
        # =========================
        cv2.imshow(WINDOW_NAME, images[current_state])

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
