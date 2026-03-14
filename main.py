from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import joblib
import shutil
import os

app = FastAPI()

# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label.joblib")


def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])

    return features.reshape(1,-1)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_location = file.filename

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    features = extract_features(file_location)

    features = scaler.transform(features)

    prediction = model.predict(features)[0]
    confidence = np.max(model.predict_proba(features))

    genre = label_encoder.inverse_transform([prediction])[0]

    os.remove(file_location)

    return {
        "filename": file.filename,
        "predicted_genre": genre,
        "confidence": float(confidence)
    }
