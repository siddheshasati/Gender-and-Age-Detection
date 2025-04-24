#Setting up Libraries
import cv2
import numpy as np
from collections import deque
import streamlit as st
from PIL import Image
import tempfile
import os

# Model files
AGE_PROTO = 'age_deploy.prototxt'
AGE_MODEL = 'age_net.caffemodel'
GENDER_PROTO = 'gender_deploy.prototxt'
GENDER_MODEL = 'gender_net.caffemodel'

# Age and gender labels
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Load models
@st.cache_resource
def load_models():
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return age_net, gender_net, face_cascade

age_net, gender_net, face_cascade = load_models()

# Preprocessing values
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def process_frame(frame, age_preds_deque, gender_preds_deque):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        if w < 80 or h < 80:
            continue  # skip small faces

        face_img = frame[y:y+h, x:x+w].copy()
        face_img = cv2.resize(face_img, (227, 227))
        face_img = cv2.GaussianBlur(face_img, (3, 3), 0)  # reduce noise

        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Gender prediction
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender_preds_deque.append(gender_preds[0])

        # Age prediction
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age_preds_deque.append(age_preds[0])

        # Average predictions
        avg_gender = np.mean(gender_preds_deque, axis=0)
        avg_age = np.mean(age_preds_deque, axis=0)

        # Confidence filter
        if avg_gender.max() < 0.5 or avg_age.max() < 0.5:
            continue  # Skip low-confidence results

        gender = GENDER_LIST[np.argmax(avg_gender)]
        age = AGE_LIST[np.argmax(avg_age)]

        label = f"{gender}, {age}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return frame

def main():
    st.title("Gender and Age Detection")
    st.sidebar.title("Options")
    
    app_mode = st.sidebar.selectbox("Choose the input mode",
                                   ["Webcam", "Upload Image"])
    
    if app_mode == "Webcam":
        st.header("Webcam Live Feed")
        st.warning("Note: This will access your webcam")
        
        run = st.checkbox('Run Webcam')
        FRAME_WINDOW = st.image([])
        
        cap = cv2.VideoCapture(0)
        
        # Store last few predictions for smoothing
        age_preds_deque = deque(maxlen=5)
        gender_preds_deque = deque(maxlen=5)
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = process_frame(frame, age_preds_deque, gender_preds_deque)
            FRAME_WINDOW.image(processed_frame)
        else:
            st.write('Stopped')
            
        cap.release()
        
    elif app_mode == "Upload Image":
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Convert PIL Image to OpenCV format
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Process the single image
            age_preds_deque = deque(maxlen=1)  # No need for smoothing for single image
            gender_preds_deque = deque(maxlen=1)
            
            processed_frame = process_frame(frame, age_preds_deque, gender_preds_deque)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            st.image(processed_frame, caption='Processed Image', use_column_width=True)
            
            # Save the processed image
            if st.button('Save Processed Image'):
                processed_pil = Image.fromarray(processed_frame)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmpfile:
                    processed_pil.save(tmpfile.name, 'JPEG')
                    st.success(f"Image saved to {tmpfile.name}")
                    with open(tmpfile.name, "rb") as file:
                        st.download_button(
                            label="Download Processed Image",
                            data=file,
                            file_name="processed_image.jpg",
                            mime="image/jpeg"
                        )
                os.unlink(tmpfile.name)

if __name__ == "__main__":
    main()
