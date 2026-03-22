from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import joblib
import shutil

app = FastAPI()

# CORS (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODEL FILES
# =========================

model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("label.joblib")
mfcc_columns = joblib.load("columns.joblib")

# =========================
# FEATURE EXTRACTION
# =========================

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=30)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = []

    for i in range(20):
        features.append(np.mean(mfcc[i]))
        features.append(np.var(mfcc[i]))

    return np.array(features).reshape(1, -1)

# =========================
# PREDICT API
# =========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_location = f"temp_{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    features = extract_mfcc(file_location)

    # convert to correct format
    import pandas as pd
    features_df = pd.DataFrame(features, columns=mfcc_columns)

    # scale
    features_scaled = scaler.transform(features_df)

    # predict
    prediction = model.predict(features_scaled)
    predicted_genre = encoder.inverse_transform(prediction)[0]

    # confidence
    probs = model.predict_proba(features_scaled)
    confidence = np.max(probs)

    return {
        "filename": file.filename,
        "predicted_genre": predicted_genre,
        "confidence": float(confidence)
    }