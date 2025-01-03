import subprocess
import sys
def install():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "package.txt"])

install()
def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

install_requirements()


import streamlit as st
import cv2
import torch
import numpy as np
from pathlib import Path

MODEL_PATH = 'yolov5/runs/train/exp42/weights/last.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True).to(device)

# Streamlit UI
st.title("Real-Time Detection with YOLOv5")
st.markdown("*Check the box below to start the camera:*")

FRAME_WINDOW = st.image([])
run = st.checkbox("Run Camera", key="run_camera")

if "cap" not in st.session_state:
    st.session_state.cap = None

if run:
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
        st.success("Camera Started!")

    while run:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to open webcam.")
            break

        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(rgb_frame)
        detections = results.pandas().xyxy[0]  
        for _, row in detections.iterrows():
            x1, y1, x2, y2, conf, cls = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['confidence'], row['class']])
            label = f"{row['name']}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    st.session_state.cap.release()
    st.session_state.cap = None
    FRAME_WINDOW.image([])
    st.checkbox("Run Camera", value=False, key="run_camera")  
else:
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        FRAME_WINDOW.image([])
    st.info("Check 'Run Camera' to start detection.")
