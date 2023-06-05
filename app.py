import streamlit as st
import requests
import json
import sounddevice as sd
import numpy as np

st.title("Real-time Speech-to-Text Transcription")

api_token = st.secrets["api_token"]
API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-base-960h"

headers = {"Authorization": f"Bearer {api_token}"}

# Initialize variables
text_transcript = ""

# Callback function to process audio data
def process_audio(indata, frames, time, status):
    global text_transcript
    audio_data = np.ravel(indata)
    response = requests.request("POST", API_URL, headers=headers, data=audio_data)
    data = json.loads(response.content.decode("utf-8"))
    text_value = data["text"]
    text_transcript += text_value

# Main app logic
if st.button("Start Transcription"):
    with sd.InputStream(callback=process_audio):
        st.write("Listening...")
        st.stop()

st.write("Transcription:")
st.write(text_transcript)
