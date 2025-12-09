import cv2
import numpy as np
import mediapipe as mp
from tensorflow.lite.python.interpreter import Interpreter

# -----------------------------
# LOAD TFLITE MODEL
# -----------------------------
MODEL_PATH = "model_numbers.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_W = input_details[0]["shape"][2]
INPUT_H = input_details[0]["shape"][1]

# -----------------------------
# MEDIAPIPE HANDS
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# -----------------------------
# GESTURE LABELS
# -----------------------------
GESTURE_LABELS = {
    0: "Idle / Stop",
    1: "Move Forward (Slow)",
    2: "Move Forward (Medium)",
    3: "Move Forward (Fast)",
    4: "Turn Left",
    5: "Turn Right"
}

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_number(roi_img):
    img = cv2.resize(roi_img, (INPUT_W, INPUT_H))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return int(np.argmax(output[0]))

# -----------------------------
# REAL-TIME WEBCAM LOOP
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    gesture = 0  # default

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Get bounding box
            x_list = [int(lm.x * w) for lm in handLms.landmark]
            y_list = [int(lm.y * h) for lm in handLms.landmark]
            x_min, x_max = max(min(x_list)-20, 0), min(max(x_list)+20, w)
            y_min, y_max = max(min(y_list)-20, 0), min(max(y_list)+20, h)
            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size != 0:
                gesture = predict_number(roi)

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

    # Display gesture
    cv2.putText(frame, f"Detected: {gesture} - {GESTURE_LABELS[gesture]}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Hand Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
