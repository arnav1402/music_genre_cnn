import streamlit as st
import requests
import os
import time

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎧", layout="centered")
st.title("🎧 Music Genre Classifier")
st.write("Upload a .mp3 or .wav file for genre prediction powered by a TensorFlow model.")

uploaded_file = st.file_uploader("🎵 Choose an audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")
    temp_path = "temp_audio." + uploaded_file.name.split('.')[-1]
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully!")

    if st.button("🎶 Predict Genre"):
        progress = st.progress(0)
        status = st.empty()

        # Step 1: Convert to Melspectrogram
        status.info('Step 1: Analysing audio and converting to Melspectrogram...')
        progress.progress(33)
        time.sleep(2)

        # Step 2: Load model
        status.info('Step 2: Loading CNN model...')
        progress.progress(66)
        time.sleep(2)

        # Step 3: Perform Prediction
        status.info('Step 3: Predicting genre...')
        try:
            with open(temp_path, "rb") as f:
                files = {"file": f}
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
            progress.progress(100)
            if response.status_code == 200:
                result = response.json()
                if "predicted_genre" in result:
                    st.success(f"🎵 Predicted Genre: **{result['predicted_genre'].title()}**")
                    st.info(f"Confidence: **{result['confidence']*100:.1f}%**")
                else:
                    st.error(result.get("error", "Unknown error"))
            else:
                st.error(f"Backend error: {response.status_code}")
        except Exception as e:
            st.error(f"Error occurred: {e}")

        status.empty()
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except PermissionError:
            st.warning("Temporary file still in use—will auto-delete later.")

