import streamlit as st
from roboflow import Roboflow
from gtts import gTTS
from PIL import Image
import io
import cv2
import tempfile
import time

# -------------------------------
# STEP 1: Initialize Roboflow
# -------------------------------
rf = Roboflow(api_key="2po24idSl5m93Vfr6ZtF")
project = rf.workspace().project("indian-currency-detection-elfyf")
model = project.version(1).model

# -------------------------------
# STEP 2: Define Browser/Cloud-friendly TTS function
# -------------------------------
def speak_currency(text):
    """Generate TTS audio using gTTS and play in Streamlit"""
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        st.audio(tmp_file.name, format="audio/mp3")

# -------------------------------
# STEP 3: Streamlit App
# -------------------------------
st.title("Indian Currency Detection with Voice Output")
st.write("Upload one or more images or use the webcam to detect currency.")

# -------------------------------
# File uploader for multiple images
# -------------------------------
uploaded_files = st.file_uploader(
    "Choose image(s)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read image
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save temporarily for Roboflow prediction
        temp_image_path = f"temp_{uploaded_file.name}"
        image.save(temp_image_path)

        # Predict and save annotated image
        result = model.predict(temp_image_path)
        result.save(f"annotated_{uploaded_file.name}")

        # Extract last detected currency
        predictions = result.json()['predictions']
        currency_detected = predictions[-1]['class'] if predictions else "Nothing"

        # Display results and play TTS
        st.image(f"annotated_{uploaded_file.name}", caption=f"Annotated: {uploaded_file.name}")
        st.write(f"Detected currency: {currency_detected}")
        speak_currency(f"The detected currency is {currency_detected}")

# -------------------------------
# Webcam capture button
# -------------------------------
if st.button("Open Camera for 10 Seconds"):
    st.write("Opening webcam...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam")
    else:
        start_time = time.time()
        last_frame = None

        while time.time() - start_time < 5:  # Capture frames for 5 seconds
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read frame from webcam")
                break
            last_frame = frame.copy()  # Keep the last frame only

        cap.release()
        st.success("Webcam capture complete")

        if last_frame is not None:
            # Save last frame temporarily
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(temp_file.name, last_frame)

            # Run prediction on the last frame
            result = model.predict(temp_file.name)
            annotated_path = f"annotated_webcam.jpg"
            result.save(annotated_path)

            # Extract last detected currency
            predictions = result.json()['predictions']
            currency_detected = predictions[-1]['class'] if predictions else "Nothing"

            # Display final annotated image and play TTS
            st.image(annotated_path, caption="Annotated Webcam Frame")
            st.write(f"Detected currency: {currency_detected}")
            speak_currency(f"The detected currency is {currency_detected}")
