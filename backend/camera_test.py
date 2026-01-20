import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Camera opened:", cap.isOpened())

while True:
    if not cap.isOpened():
        break

    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Raw Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    # ESC key
    if key == 27:
        break

    # Window close (X button)
    if cv2.getWindowProperty("Raw Camera Feed", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("Camera released cleanly")
