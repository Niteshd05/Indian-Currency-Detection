import streamlit as st
from roboflow import Roboflow
from PIL import Image
import io
import tempfile
import requests
import os

# -------------------------------
# STEP 1: Initialize Roboflow
# -------------------------------
rf = Roboflow(api_key="2po24idSl5m93Vfr6ZtF")
project = rf.workspace().project("indian-currency-detection-elfyf")
model = project.version(1).model

# -------------------------------
# STEP 2: ElevenLabs TTS Function
# -------------------------------
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") # ⬅️ Replace with your real API key
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Default voice (Adam)

def speak_currency(text):
    """Play ElevenLabs TTS audio with autoplay using HTML"""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "text": text,
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.8
        }
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        audio_bytes = response.content
        audio_base64 = audio_bytes.encode("base64") if hasattr(audio_bytes, "encode") else None
        import base64
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    else:
        st.error(f"Error: ElevenLabs API returned {response.status_code}")



# -------------------------------
# STEP 3: Streamlit App
# -------------------------------
st.title("💵 Indian Currency Detection with ElevenLabs Voice Output")
st.write("Upload one or more images or use the webcam to detect currency.")

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

