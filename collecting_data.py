import cv2
import mediapipe as mp
import time
import pickle
import tkinter as tk
from tkinter import simpledialog
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1  
)

root = tk.Tk()
root.withdraw() 

def get_gesture_name():
    gesture_name = simpledialog.askstring("Input", "Enter gesture name:")
    if gesture_name:
        return gesture_name
    else:
        return None

gesture_data = []
gesture_labels = []

cap = cv2.VideoCapture(0)

while cap.isOpened():
    current_gesture = get_gesture_name()

    if current_gesture is None:
        print("Terminating capture...")
        cap.release()
        cv2.destroyAllWindows()
        break

    gesture_data.clear()
    gesture_labels.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                hand_coords = []
                for landmark in hand_landmarks.landmark:
                    hand_coords.extend([landmark.x, landmark.y, landmark.z])

                if hand_coords:
                    gesture_data.append(hand_coords)
                    gesture_labels.append(current_gesture)

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            print(f"Capturing gesture: {current_gesture}")
            if len(gesture_data) > 0:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                file_name = f"gestures_{current_gesture}_{timestamp}.pkl"
                with open(file_name, 'wb') as f:
                    pickle.dump((gesture_data, gesture_labels), f)
                print(f"Data saved for gesture: {current_gesture} in {file_name}")

            current_gesture = get_gesture_name()

            if current_gesture is None:
                print("Terminating capture...")
                cap.release()
                cv2.destroyAllWindows()
                break

cap.release()
cv2.destroyAllWindows()

def load_data():
    gesture_data = []
    gesture_labels = []

    for file in os.listdir():
        if file.endswith(".pkl"):
            with open(file, 'rb') as f:
                try:
                    data, labels = pickle.load(f)
                    gesture_data.extend(data)
                    gesture_labels.extend(labels)
                except TypeError:
                    print(f"Error loading {file}: File may contain a single object.")
                    continue

    return gesture_data, gesture_labels

gesture_data, gesture_labels = load_data()

print(f"Data loaded: {len(gesture_data)} samples")
