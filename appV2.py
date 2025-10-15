import streamlit as st
from roboflow import Roboflow
import pyttsx3
from PIL import Image
import io
import tempfile

# -------------------------------
# STEP 1: Initialize Roboflow
# -------------------------------
rf = Roboflow(api_key="2po24idSl5m93Vfr6ZtF")
project = rf.workspace().project("indian-currency-detection-elfyf")
model = project.version(1).model

# -------------------------------
# STEP 2: Define TTS function
# -------------------------------
def speak_currency(text):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for v in voices:
            if "female" in v.name.lower() or "female" in str(v.gender).lower():
                engine.setProperty('voice', v.id)
                break
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        st.warning(f"Text-to-speech failed: {str(e)}")

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

        # Save temporarily for Roboflow
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        # Predict and save annotated image
        result = model.predict(temp_path)
        annotated_path = f"annotated_{uploaded_file.name}"
        result.save(annotated_path)

        # Extract last detected currency
        predictions = result.json()['predictions']
        currency_detected = predictions[-1]['class'] if predictions else "Nothing"

        # Display results and speak
        st.image(annotated_path, caption=f"Annotated: {uploaded_file.name}")
        st.write(f"Detected currency: {currency_detected}")
        speak_currency(f"The detected currency is {currency_detected}")

# -------------------------------
# Webcam capture using Streamlit camera_input
# -------------------------------
st.write("---")
st.header("Use Webcam to Detect Currency")

img_file = st.camera_input("Take a picture")
if img_file is not None:
    # Read image
    image = Image.open(img_file)

    # Save temporarily for Roboflow
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    # Predict and save annotated image
    result = model.predict(temp_path)
    annotated_path = "annotated_webcam.jpg"
    result.save(annotated_path)

    # Extract last detected currency
    predictions = result.json()['predictions']
    currency_detected = predictions[-1]['class'] if predictions else "Nothing"

    # Display final annotated image and speak
    st.image(annotated_path, caption="Annotated Webcam Image")
    st.write(f"Detected currency: {currency_detected}")
    speak_currency(f"The detected currency is {currency_detected}")
