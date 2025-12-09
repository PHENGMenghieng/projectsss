import cv2
import mediapipe as mp
import os

# ------------------------------
# CONFIG
# ------------------------------
NUM_CLASSES = 6        # 0–5
SAMPLES_PER_CLASS = 300
SAVE_DIR = "dataset"   # All images will be saved here

# Create directories
for i in range(NUM_CLASSES):
    path = os.path.join(SAVE_DIR, str(i))
    os.makedirs(path, exist_ok=True)

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as hands:

    for number in range(NUM_CLASSES):
        print(f"Collect images for number: {number}")
        count = 0
        while count < SAMPLES_PER_CLASS:
            ret, frame = cap.read()
            if not ret:
                continue

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                    # Bounding box
                    x_list = [int(lm.x * w) for lm in handLms.landmark]
                    y_list = [int(lm.y * h) for lm in handLms.landmark]

                    x_min, x_max = max(min(x_list) - 20, 0), min(max(x_list) + 20, w)
                    y_min, y_max = max(min(y_list) - 20, 0), min(max(y_list) + 20, h)

                    roi = frame[y_min:y_max, x_min:x_max]

                    if roi.size != 0:
                        # Save image
                        filename = os.path.join(SAVE_DIR, str(number), f"{count}.jpg")
                        cv2.imwrite(filename, roi)
                        count += 1
                        print(f"Saved {filename}")

            cv2.imshow("Collecting dataset", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
