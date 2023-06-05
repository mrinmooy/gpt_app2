import streamlit as st
import speech_recognition as sr

def main():
    st.title("Real-time Speech-to-Text Transcription")

    r = sr.Recognizer()

    with sr.Microphone() as source:
        button_start = st.button("🎙️")

        if button_start:
            st.info("Listening... Say something!")
            audio = r.listen(source)

            try:
                text = r.recognize_google(audio)
                st.success(f"You said: {text}")
            except sr.UnknownValueError:
                st.warning("Sorry, I could not understand what you said.")
            except sr.RequestError as e:
                st.error(f"Request error: {e}")

if __name__ == '__main__':
    main()
