import cv2
import mediapipe as mp
import torch
import numpy as np
import pygame
from gtts import gTTS
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import os

class GestureRecognitionNN(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureRecognitionNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, num_classes)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

gesture_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
label_encoder = LabelEncoder()
label_encoder.fit(gesture_labels)

input_size = 63
num_classes = len(gesture_labels)

model = GestureRecognitionNN(input_size, num_classes)
model.load_state_dict(torch.load("gesture_recognition_model.pth"))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

pygame.mixer.init()

last_gesture = None

def predict_gesture(gesture_data):
    global last_gesture

    with torch.no_grad():
        gesture_data = torch.tensor(gesture_data, dtype=torch.float32).reshape(1, -1)
        output = model(gesture_data)
        _, predicted = torch.max(output, 1)
        predicted_label = label_encoder.inverse_transform([predicted.item()])
        
        gesture_name = predicted_label[0]

        if gesture_name != last_gesture:
            last_gesture = gesture_name

            pygame.mixer.music.stop()
            pygame.mixer.quit()
            pygame.mixer.init()

            tts = gTTS(text=gesture_name, lang='en')
            temp_audio_path = "gesture_name.mp3"

            try:
                os.remove(temp_audio_path)
            except FileNotFoundError:
                pass
            except PermissionError:
                print(f"Could not delete {temp_audio_path}. It may be in use.")

            tts.save(temp_audio_path)

            pygame.mixer.music.load(temp_audio_path)
            pygame.mixer.music.play()

        return gesture_name


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_coords = []
            for landmark in hand_landmarks.landmark:
                hand_coords.extend([landmark.x, landmark.y, landmark.z])

            predicted_gesture = predict_gesture(hand_coords)
            cv2.putText(frame, f"Gesture: {predicted_gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
