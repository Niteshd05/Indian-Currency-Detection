import streamlit as st
from roboflow import Roboflow
from PIL import Image
import io
import tempfile
import urllib.parse

# -------------------------------
# STEP 1: Initialize Roboflow
# -------------------------------
rf = Roboflow(api_key="2po24idSl5m93Vfr6ZtF")
project = rf.workspace().project("indian-currency-detection-elfyf")
model = project.version(1).model

# -------------------------------
# STEP 2: Define browser-based TTS function
# -------------------------------
def speak_currency(text):
    """Use Google Translate TTS via HTML audio tag"""
    text_encoded = urllib.parse.quote(text)
    st.markdown(f"""
    <audio autoplay>
      <source src="https://translate.google.com/translate_tts?ie=UTF-8&q={text_encoded}&tl=en&client=tw-ob" type="audio/mpeg">
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
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        result = model.predict(temp_path)
        annotated_path = f"annotated_{uploaded_file.name}"
        result.save(annotated_path)

        predictions = result.json()['predictions']
        currency_detected = predictions[-1]['class'] if predictions else "Nothing"

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
    image = Image.open(img_file)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    result = model.predict(temp_path)
    annotated_path = "annotated_webcam.jpg"
    result.save(annotated_path)

    predictions = result.json()['predictions']
    currency_detected = predictions[-1]['class'] if predictions else "Nothing"

    st.image(annotated_path, caption="Annotated Webcam Image")
    st.write(f"Detected currency: {currency_detected}")
    speak_currency(f"The detected currency is {currency_detected}")
