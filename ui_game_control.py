import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
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
# MEDIAPIPE INIT (MOVE UP HERE!)
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# -----------------------------
# PREDICT FUNCTION
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
# TKINTER UI
# -----------------------------
root = tk.Tk()
root.title("Gesture Controlled UI Game")
root.geometry("900x600")
root.configure(bg="#222")

camera_label = tk.Label(root, bg="#222")
camera_label.pack(side="left", padx=20, pady=20)

canvas = tk.Canvas(root, width=400, height=400, bg="#333", highlightthickness=0)
canvas.pack(side="right", padx=20)

player = canvas.create_rectangle(180, 180, 220, 220, fill="#00d4ff")

gesture_text = tk.Label(root, text="Gesture: --", fg="white", bg="#222", font=("Arial", 20))
gesture_text.pack(pady=10)

VEL = 10
jump = False
jump_speed = 0

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

def update_frame():
    global jump, jump_speed

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    # Better performance: resize before processing
    frame_small = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)
    gesture = 0

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame_small, handLms, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame_small.shape
            x_list = [int(lm.x * w) for lm in handLms.landmark]
            y_list = [int(lm.y * h) for lm in handLms.landmark]

            x_min, x_max = max(min(x_list) - 20, 0), min(max(x_list) + 20, w)
            y_min, y_max = max(min(y_list) - 20, 0), min(max(y_list) + 20, h)

            roi = frame_small[y_min:y_max, x_min:x_max]

            if roi.size != 0:
                gesture = predict_number(roi)

    # Update UI label
    gesture_text.config(text=f"Gesture: {gesture}")

    # -----------------------------
    # GAME LOGIC
    # -----------------------------
    x1, y1, x2, y2 = canvas.coords(player)

    if gesture == 1:
        canvas.move(player, -VEL, 0)
    elif gesture == 2:
        canvas.move(player, +VEL, 0)
    elif gesture == 3:
        if not jump:
            jump = True
            jump_speed = -15
    elif gesture == 4:
        canvas.move(player, 0, +VEL)
    elif gesture == 5:
        canvas.create_rectangle(x2, y1+8, x2+20, y1+12, fill="red")

    if jump:
        canvas.move(player, 0, jump_speed)
        jump_speed += 1
        if y2 >= 400:
            jump = False

    # Display camera in Tkinter
    img = ImageTk.PhotoImage(Image.fromarray(rgb))
    camera_label.imgtk = img
    camera_label.configure(image=img)

    root.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()