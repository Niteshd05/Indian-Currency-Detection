import streamlit as st
from roboflow import Roboflow
from PIL import Image
import io
import tempfile
import time

# -------------------------------
# STEP 1: Initialize Roboflow
# -------------------------------
rf = Roboflow(api_key="2po24idSl5m93Vfr6ZtF")
project = rf.workspace().project("indian-currency-detection-elfyf")
model = project.version(1).model

# -------------------------------
# STEP 2: Browser-based TTS
# -------------------------------
def speak_currency(text):
    # Use Google TTS in the browser
    st.markdown(f"""
    <audio autoplay>
      <source src="https://translate.google.com/translate_tts?ie=UTF-8&q={text}&tl=en&client=tw-ob" type="audio/mpeg">
    </audio>
    """, unsafe_allow_html=True)

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

        # Display results and speak
        st.image(f"annotated_{uploaded_file.name}", caption=f"Annotated: {uploaded_file.name}")
        st.write(f"Detected currency: {currency_detected}")
        speak_currency(f"The detected currency is {currency_detected}")

# -------------------------------
# Browser-based webcam capture
# -------------------------------
st.write("---")
st.header("Use Webcam to Detect Currency")

img_file = st.camera_input("Take a picture")
if img_file is not None:
    # Read image
    image = Image.open(img_file)
    
    # Save temporarily for Roboflow
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    image.save(temp_file.name)

    # Run prediction
    result = model.predict(temp_file.name)
    annotated_path = "annotated_webcam.jpg"
    result.save(annotated_path)

    # Extract last detected currency
    predictions = result.json()['predictions']
    currency_detected = predictions[-1]['class'] if predictions else "Nothing"

    # Display final annotated image and speak
    st.image(annotated_path, caption="Annotated Webcam Image")
    st.write(f"Detected currency: {currency_detected}")
    speak_currency(f"The detected currency is {currency_detected}")
