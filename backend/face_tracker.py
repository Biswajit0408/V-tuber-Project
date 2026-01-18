import cv2
import mediapipe as mp
import math

# ----------------------------
# Utility functions
# ----------------------------
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


# ----------------------------
# MediaPipe setup
# ----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------
# Webcam
# ----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
WINDOW_NAME = "VTuber Face Tracking - Phase 2"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# ----------------------------
# Main loop
# ----------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            # Eye distances
            left_eye = distance(points[159], points[145])
            right_eye = distance(points[386], points[374])

            # Mouth open
            mouth_open = distance(points[13], points[14])

            # Head yaw
            nose_x = points[1][0]
            left_face = points[234][0]
            right_face = points[454][0]
            face_center = (left_face + right_face) // 2
            head_yaw = nose_x - face_center

            # Display values
            cv2.putText(frame, f"Left Eye: {left_eye:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Right Eye: {right_eye:.2f}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Mouth Open: {mouth_open:.2f}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Head Yaw: {head_yaw}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released cleanly")
