import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class GestureDataset(Dataset):
    def __init__(self, gesture_data, gesture_labels, transform=None):
        self.gesture_data = gesture_data
        self.gesture_labels = gesture_labels
        self.transform = transform

        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(gesture_labels)

    def __len__(self):
        return len(self.gesture_data)

    def __getitem__(self, idx):
        data = np.array(self.gesture_data[idx], dtype=np.float32)
        label = self.encoded_labels[idx]
        
        data = data.reshape(-1)
        
        if self.transform:
            data = self.transform(data)
        
        return torch.tensor(data), torch.tensor(label)

def load_data():
    gesture_data = []
    gesture_labels = []
    
    for file in os.listdir():
        if file.endswith(".pkl"):
            with open(file, 'rb') as f:
                data, labels = pickle.load(f)
                gesture_data.extend(data)
                gesture_labels.extend(labels)
    
    return gesture_data, gesture_labels

class GestureRecognitionNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureRecognitionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

gesture_data, gesture_labels = load_data()
dataset = GestureDataset(gesture_data, gesture_labels)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

input_size = len(gesture_data[0])  
num_classes = len(set(gesture_labels))  

model = GestureRecognitionNN(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
    
    accuracy = 100 * correct_predictions / total_predictions
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "gesture_recognition_model.pth")

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(dataset.label_encoder, f)

print("Training completed and model saved.")
