import streamlit as st
from roboflow import Roboflow
from PIL import Image
import io
import tempfile
import requests
import os
from gtts import gTTS

# -------------------------------
# STEP 1: Load ElevenLabs API Key
# -------------------------------
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Default free voice ("Adam")

# -------------------------------
# STEP 2: Roboflow Model Setup
# -------------------------------
rf = Roboflow(api_key="2po24idSl5m93Vfr6ZtF")
project = rf.workspace().project("indian-currency-detection-elfyf")
model = project.version(1).model

# -------------------------------
# STEP 3: TTS Function with Silent Fallback
# -------------------------------
def speak_currency(text):
    """Try ElevenLabs first, silently fallback to gTTS if unavailable"""
    audio_played = False
    if ELEVENLABS_API_KEY:
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVENLABS_API_KEY
            }
            payload = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.4,
                    "similarity_boost": 0.8
                }
            }
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                st.audio(response.content, format="audio/mp3")
                audio_played = True
        except:
            pass  # silently ignore any ElevenLabs errors

    if not audio_played:
        tts = gTTS(text)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        st.audio(audio_fp, format="audio/mp3")

# -------------------------------
# STEP 4: Streamlit UI
# -------------------------------
st.title("💵 Indian Currency Detection with TTS")
st.write("Upload images or use webcam to detect Indian currency.")

# -------------------------------
# File uploader
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
        st.write(f"💰 Detected currency: **{currency_detected}**")
        speak_currency(f"The detected currency is {currency_detected}")

# -------------------------------
# Webcam capture
# -------------------------------
st.write("---")
st.header("📸 Use Webcam to Detect Currency")

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
    st.write(f"💰 Detected currency: **{currency_detected}**")
    speak_currency(f"The detected currency is {currency_detected}")
