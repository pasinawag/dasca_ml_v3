import cv2
import mediapipe as mp
import numpy as np
import torch

# Define the model class
class LandmarkClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LandmarkClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Load the trained model
input_size = 42  # Set this to match the training configuration
hidden_size = 128
num_classes = 5  # Adjust based on your actual number of classes
model = LandmarkClassifier(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('landmark_model.pth'))
model.eval()  # Set the model to evaluation mode

cap = cv2.VideoCapture(1)

# Mediapipe Hands initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Labels for predictions
labels_dict = {0: 'STEP 1: CORRECT', 1: 'STEP 2: CORRECT', 2: 'STEP 3: CORRECT', 3: 'STEP 4: CORRECT', 4: 'Hindi: CORRECT'}

# Function to pad or truncate data to ensure the correct feature size
def pad_or_truncate(data, max_length):
    if len(data) < max_length:
        # Pad with zeros if data is shorter
        data += [0] * (max_length - len(data))
    else:
        # Truncate if data is longer
        data = data[:max_length]
    return data

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Mediapipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,  # image to draw on
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Extract landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Ensure feature vector has correct length (42, adjust this if necessary)
        data_aux = pad_or_truncate(data_aux, 42)

        # Convert to PyTorch tensor
        data_tensor = torch.tensor(data_aux, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():  # No gradients needed during inference
            outputs = model(data_tensor)
            probabilities = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            max_prob, predicted_class = torch.max(probabilities, dim=1)

        # Check if the maximum probability exceeds the threshold (64% or 0.64)
        if max_prob.item() >= 0.64:
            predicted_character = labels_dict[int(predicted_class.item())]
        else:
            predicted_character = 'UNDETERMINED'

        # Calculate bounding box for drawing
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Draw bounding box and prediction text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
