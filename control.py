# import cv2
# import mediapipe as mp

# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils

# # Finger tip landmark IDs
# TIP_IDS = [4, 8, 12, 16, 20]

# def count_fingers(hand_landmarks):
#     fingers = []

#     # Thumb (special case: compare x-coordinate)
#     if hand_landmarks.landmark[TIP_IDS[0]].x < hand_landmarks.landmark[TIP_IDS[0] - 1].x:
#         fingers.append(1)
#     else:
#         fingers.append(0)

#     # Other 4 fingers (compare y-coordinate)
#     for id in range(1, 5):
#         if hand_landmarks.landmark[TIP_IDS[id]].y < hand_landmarks.landmark[TIP_IDS[id] - 2].y:
#             fingers.append(1)
#         else:
#             fingers.append(0)

#     return fingers.count(1)


# cap = cv2.VideoCapture(0)

# with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(rgb)

#         if results.multi_hand_landmarks:
#             handLms = results.multi_hand_landmarks[0]
#             mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

#             # Count raised fingers
#             finger_count = count_fingers(handLms)

#             # Map finger count to robot actions
#             if finger_count == 1:
#                 action = "MOVE FORWARD"
#             elif finger_count == 2:
#                 action = "MOVE BACKWARD"
#             elif finger_count == 3:
#                 action = "TURN AROUND"
#             elif finger_count == 0:
#                 action = "STOP"
#             elif finger count == 4:
#                 action = 
#             else:
#                 action = "NO ACTION"

#             cv2.putText(frame, f"Fingers: {finger_count}", (30, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

#             cv2.putText(frame, f"Action: {action}", (30, 110),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 3)

#             print("Finger count:", finger_count, "->", action)

#         cv2.imshow("Hand Control", frame)
#         if cv2.waitKey(1) == 27:  
#             break

# cap.release()
# cv2.destroyAllWindows()

# main.py (Conceptual Update)

# import cv2
# import mediapipe as mp
# import numpy as np
# # Import TFLite interpreter
# from tensorflow.lite import Interpreter


# # Import the robot control functions
# from bot import ROBOT_ACTIONS 
# # Note: Ensure bot.py is in the same directory

# # --- 1. SETUP ---
# # Load TFLite Model
# MODEL_PATH = 'your_model_path.tflite' # <--- CHANGE THIS PATH!
# interpreter = Interpreter(model_path=MODEL_PATH)
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# INPUT_SHAPE = input_details[0]['shape'][1:3] # Should be (224, 224)

# # MediaPipe Setup
# mp_hands = mp.solutions.hands
# # ... (mp_draw setup as before)

# def preprocess_and_predict(hand_img_roi):
#     """Crops, resizes, normalizes, and runs TFLite inference."""
#     # Resize to 224x224 and normalize to [0, 1]
#     input_tensor = cv2.resize(hand_img_roi, INPUT_SHAPE)
#     input_tensor = input_tensor.astype('float32') / 255.0
#     input_tensor = np.expand_dims(input_tensor, axis=0) # Add batch dimension

#     # Run inference
#     interpreter.set_tensor(input_details[0]['index'], input_tensor)
#     interpreter.invoke()
    
#     # Get the prediction
#     output = interpreter.get_tensor(output_details[0]['index'])
#     predicted_number = np.argmax(output[0])
    
#     return predicted_number

# # --- 2. MAIN LOOP (Simplified for clarity) ---
# cap = cv2.VideoCapture(0)

# with mp_hands.Hands(...) as hands:
#     while True:
#         ret, frame = cap.read()
#         # ... (MediaPipe detection and processing as before)

#         if results.multi_hand_landmarks:
#             # --- GET ROI (Region of Interest) ---
#             # You would use MediaPipe landmarks here to calculate 
#             # the bounding box and crop the hand ROI from the frame.
#             # This is complex and needs to be implemented accurately.
            
#             # Placeholder for the cropped hand image (BGR format)
#             hand_roi = frame[y_min:y_max, x_min:x_max] 
            
#             if hand_roi.size != 0:
#                 # --- PREDICT NUMBER ---
#                 predicted_number = preprocess_and_predict(hand_roi)
                
#                 # --- EXECUTE ROBOT ACTION ---
#                 action_func = ROBOT_ACTIONS.get(predicted_number)
#                 if action_func:
#                     action_func() # Calls the function in bot.py
#                     action_name = action_func.__name__
#                 else:
#                     action_name = "ERROR: Invalid Number"

#                 # ... (cv2.putText for display as before, using action_name)



import cv2
import numpy as np
import mediapipe as mp

# ✔ Correct TFLite Interpreter import
from tensorflow.lite.python.interpreter import Interpreter

# -----------------------------
# CONFIG
# -----------------------------

MODEL_PATH = "model_numbers.tflite"     # <--- put your TFLite model here

# Load TFLite model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_W = input_details[0]["shape"][2]
INPUT_H = input_details[0]["shape"][1]

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# ROBOT ACTIONS
# -----------------------------

def do_stop():
    print("ACTION: STOP (0)")

def forward_slow():
    print("ACTION: FORWARD SLOW (1)")

def forward_med():
    print("ACTION: FORWARD MEDIUM (2)")

def forward_fast():
    print("ACTION: FORWARD FAST (3)")

def turn_left():
    print("ACTION: TURN LEFT (4)")

def turn_right():
    print("ACTION: TURN RIGHT (5)")

ACTIONS = {
    0: do_stop,
    1: forward_slow,
    2: forward_med,
    3: forward_fast,
    4: turn_left,
    5: turn_right
}

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------

def predict_number(roi_img):
    """Resize ROI to model size, normalize, run inference"""
    img = cv2.resize(roi_img, (INPUT_W, INPUT_H))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return int(np.argmax(output[0]))


# -----------------------------
# MAIN LOOP
# -----------------------------

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                # -----------------------------
                # Extract bounding box
                # -----------------------------
                x_list = []
                y_list = []

                for lm in handLms.landmark:
                    x_list.append(int(lm.x * w))
                    y_list.append(int(lm.y * h))

                x_min, x_max = max(min(x_list) - 20, 0), min(max(x_list) + 20, w)
                y_min, y_max = max(min(y_list) - 20, 0), min(max(y_list) + 20, h)

                roi = frame[y_min:y_max, x_min:x_max]

                if roi.size != 0:
                    number = predict_number(roi)

                    # Perform robot action
                    action_func = ACTIONS.get(number)
                    if action_func:
                        action_func()

                    # Display
                    cv2.putText(frame, f"Detected: {number}",
                                (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)

        cv2.imshow("Number Gesture Robot", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
