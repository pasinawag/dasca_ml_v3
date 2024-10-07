import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder  # Import for label encoding

# Function to pad or truncate sequences to a fixed length
def pad_or_truncate(data, max_length):
    padded_data = []
    for sequence in data:
        if len(sequence) < max_length:
            padded_sequence = sequence + [0] * (max_length - len(sequence))
        else:
            padded_sequence = sequence[:max_length]
        padded_data.append(padded_sequence)
    return np.array(padded_data)

# PyTorch dataset class to load and prepare the data
class LandmarkDataset(Dataset):
    def __init__(self, data, labels, max_length):
        self.labels = labels
        self.landmarks = pad_or_truncate(data, max_length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.landmarks[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# Load the data
data_dict = pickle.load(open('./data_no.pickle', 'rb'))

# Assuming data_dict['data'] contains the landmarks and data_dict['labels'] contains the labels
data = data_dict['data']
labels = data_dict['labels']  # Keeping it as a list for now

# Convert labels from strings to integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Find the length of the longest sequence
max_length = max(len(seq) for seq in data)

# Create dataset and dataloader
dataset = LandmarkDataset(data, labels_encoded, max_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Hyperparameters
input_size = max_length
hidden_size = 128
num_classes = len(np.unique(labels_encoded))  # Use encoded labels for unique class count
learning_rate = 0.001
num_epochs = 100

# Define the model
class LandmarkClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LandmarkClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Model, loss function, and optimizer
model = LandmarkClassifier(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'landmark_model.pth')
