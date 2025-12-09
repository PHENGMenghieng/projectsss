import cv2
import numpy as np
import mediapipe as mp
from tensorflow.lite.python.interpreter import Interpreter

# --- Load model ---
MODEL_PATH = "model_numbers.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_W = input_details[0]["shape"][2]
INPUT_H = input_details[0]["shape"][1]

# --- Meaning of numbers ---
GESTURE_LABELS = {
    0: "Idle / Stop",
    1: "Move Forward (Slow)",
    2: "Move Forward (Medium)",
    3: "Move Forward (Fast)",
    4: "Turn Left",
    5: "Turn Right"
}

# --- Mediapipe hand detection ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def predict_number(roi_img):
    img = cv2.resize(roi_img, (INPUT_W, INPUT_H))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)   # shape: (1, H, W, 3)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return int(np.argmax(output[0]))


# --- Load your test picture ---
image_path = r"C:\Users\TGS\OneDrive\Pictures\Camera Roll\WIN_20251209_13_50_20_Pro.jpg"
img = cv2.imread(image_path)

if img is None:
    print("❌ Could not load image. Check file name.")
    exit()

# Detect hand using Mediapipe
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = hands.process(rgb)

if not results.multi_hand_landmarks:
    print("❌ No hand detected!")
    exit()

# Extract hand ROI
h, w, _ = img.shape
hand = results.multi_hand_landmarks[0]

x_list = [int(lm.x * w) for lm in hand.landmark]
y_list = [int(lm.y * h) for lm in hand.landmark]

x_min, x_max = max(min(x_list) - 50, 0), min(max(x_list) + 50, w)
y_min, y_max = max(min(y_list) - 50, 0), min(max(y_list) + 50, h)

roi = img[y_min:y_max, x_min:x_max]

cv2.imshow("ROI Used For Prediction", roi)

prediction = predict_number(roi)
meaning = GESTURE_LABELS.get(prediction, "Unknown")

print("Detected Number:", prediction)
print("Meaning:", meaning)

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
