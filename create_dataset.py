import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Path to dataset
DATA_DIR = r'C:\Users\hp\Downloads\sign-language-detector-python-master\data'

if not os.path.exists(DATA_DIR):
    print(f"The directory {DATA_DIR} does not exist.")
    exit()

data = []
labels = []

# Loop through dataset folders
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):  # Ensure it's a directory
        for img_path in os.listdir(dir_path):
            if img_path.endswith(('.jpg', '.jpeg', '.png')):  # Valid image check
                data_aux = []
                x_ = []
                y_ = []

                try:
                    img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"Error processing image {img_path} in {dir_}: {e}")
                    continue

                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Collect x and y coordinates
                        for lm in hand_landmarks.landmark:
                            x_.append(lm.x)
                            y_.append(lm.y)

                        # Normalize landmarks
                        for lm in hand_landmarks.landmark:
                            data_aux.append(lm.x - min(x_))
                            data_aux.append(lm.y - min(y_))

                    # Save sample
                    data.append(data_aux)
                    labels.append(dir_)

# Save dataset to pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data has been saved to data.pickle successfully.")
