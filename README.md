# 🎵 Music Genre Classifier

A CNN-based deep learning model with **70K+ parameters** built with TensorFlow that classifies audio tracks by genre in real-time. The trained model is served via a **FastAPI** backend and a **Streamlit** frontend UI.

---

## 🧠 Project Overview

- Extracts audio features (MFCCs, Mel Spectrograms) from raw audio files
- Trains a Convolutional Neural Network (CNN) using TensorFlow/Keras
- Serves predictions through a RESTful FastAPI API
- Provides an interactive Streamlit UI for real-time genre prediction

---

## 📁 Project Structure

```
music-genre-classifier/
│
├── train.ipynb          # Jupyter notebook to train and export the CNN model
├── model/
│   └── genre_model.h5   # Saved trained model (generated after training)
│
├── api.py               # FastAPI app — loads model and exposes prediction endpoints
│
├── app.py               # Streamlit UI — upload audio and get genre predictions
│
├── data/                # Place your training audio dataset here
└── README.md
```

---

## ⚙️ Environment Setup (Conda + CUDA)

> Requires: [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and a CUDA-compatible GPU.

### 1. Create the Conda Environment

```bash
conda create -n music-genre python=3.10 -y
conda activate music-genre
```

### 2. Install CUDA Toolkit via Conda

```bash
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.6 -y
```

### 3. Install Python Dependencies

```bash
pip install tensorflow[and-cuda]==2.13.0
pip install pandas numpy librosa scikit-learn matplotlib
pip install fastapi uvicorn python-multipart
pip install streamlit requests
conda activate music-genre
```

### 4. Verify GPU Detection

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

You should see your GPU listed. If the list is empty, double-check your CUDA installation.

---

## 🏋️ Training the Model

Open and run the training notebook:

```bash
conda activate music-genre
jupyter notebook train.ipynb
```

Run all cells in `train.ipynb`. This will:

1. Load and preprocess audio files from `data/`
2. Extract MFCC / Mel Spectrogram features
3. Train the CNN model (~70K+ parameters)
4. Save the trained model to `model/genre_model.h5`

> Make sure your dataset is placed inside the `data/` directory before running the notebook. The [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) is recommended.

---

## 🚀 Running the Application

You will need **two terminals**, both with the Conda environment activated.

---

### Terminal 1 — Start the FastAPI Backend

```bash
conda activate music-genre
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be live at: `http://localhost:8000`

Interactive API docs (Swagger UI): `http://localhost:8000/docs`

**Key endpoint:**

| Method | Endpoint      | Description                        |
|--------|---------------|------------------------------------|
| POST   | `/predict`    | Upload an audio file → get genre   |

---

### Terminal 2 — Start the Streamlit UI

```bash
conda activate music-genre
cd app/
streamlit run app.py
```

The UI will be live at: `http://localhost:8501`

Upload any `.wav` or `.mp3` file and get an instant genre prediction powered by the FastAPI backend.

---

## 🔌 API Usage Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio_file.wav"
```

**Response:**

```json
{
  "genre": "jazz",
  "confidence": 0.94
}
```
## 🎯 Supported Genres

The model is trained to classify the following genres (GTZAN):

`blues` · `classical` · `country` · `disco` · `hiphop` · `jazz` · `metal` · `pop` · `reggae` · `rock`

---

## 🛠️ Tech Stack

| Component     | Technology                        |
|---------------|-----------------------------------|
| Model         | TensorFlow / Keras (CNN)          |
| Audio Features| Librosa (MFCCs, Mel Spectrograms) |
| Backend API   | FastAPI + Uvicorn                 |
| Frontend UI   | Streamlit                         |
| Environment   | Conda + CUDA 11.8                 |
| Language      | Python 3.10                       |

- Ensure `model/genre_model.h5` exists before starting the API (run `train.ipynb` first)
- Audio files should ideally be at least 3 seconds long for accurate predictions
- The API and Streamlit app must both be running at the same time for the UI to work
