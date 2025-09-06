import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        r"C:\Users\hp\Downloads\project DL 2\project DL - Copy (2)\sign-language-detector-python-master\model_mlp.h5"
    )

model = load_model()

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels for each class
labels_dict = {
    0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 
    5: '6', 6: '7', 7: '8', 8: '9', 9: '10'
}

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        color: #ffffff;
        background-color: #3B4A6B;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #2D3B54;
    }
    .status-bar {
        color: #ffffff;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
    }
    .status-error {
        background-color: #D9534F;
    }
    .status-success {
        background-color: #5CB85C;
    }
    .frame-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.title("✋ Hand Gesture Recognition System")
st.markdown("An elegant application to detect numbers in sign language using a deep learning model (MLP).")

col1, col2 = st.columns([1, 1])
with col1:
    start_button = st.button("Start Video")
with col2:
    stop_button = st.button("Stop Video")

frame_placeholder = st.empty()
status_placeholder = st.empty()

if start_button:
    cap = cv2.VideoCapture(0)
    status_placeholder.markdown("<div class='status-bar status-success'>Video capture started successfully!</div>", unsafe_allow_html=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            status_placeholder.markdown("<div class='status-bar status-error'>Error: Could not read frame.</div>", unsafe_allow_html=True)
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux = []
        x_ = []
        y_ = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            if len(data_aux) == 42:  # MLP expects (1, 42)
                data_aux = np.asarray(data_aux).reshape(1, 42)

                prediction = model.predict(data_aux)
                predicted_label = int(np.argmax(prediction, axis=-1)[0])
                predicted_character = labels_dict.get(predicted_label, "Unknown")

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (59, 72, 107), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (235, 235, 235), 3, cv2.LINE_AA)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)

        with frame_placeholder.container():
            st.image(frame_image, caption="Live Video Feed", use_column_width=True)

        if stop_button:
            cap.release()
            status_placeholder.markdown("<div class='status-bar status-error'>Video capture stopped.</div>", unsafe_allow_html=True)
            break

    cap.release()
else:
    status_placeholder.markdown("<div class='status-bar status-error'>Video capture is inactive. Press 'Start' to begin.</div>", unsafe_allow_html=True)

cv2.destroyAllWindows()
