# api.py
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.image import resize
import numpy as np
import librosa
import os

app = FastAPI()

MODEL_PATH = "./trained_model.h5"
model = load_model(MODEL_PATH)

CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz']

def predict_genre_from_file(model, file_path, classes, target_shape=(150, 150)):
    # Load audio
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    # Define chunking parameters
    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    data = []

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    data = np.array(data)
    predictions = model.predict(data)
    avg_pred = np.mean(predictions, axis=0)
    predicted_class = classes[np.argmax(avg_pred)]
    return predicted_class, float(np.max(avg_pred))


@app.post("/predict")
async def predict_genre(file: UploadFile):
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        predicted_genre, confidence = predict_genre_from_file(model, temp_path, CLASSES)
        return JSONResponse({
            "predicted_genre": predicted_genre,
            "confidence": round(confidence, 3)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
