import librosa
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATASET_PATH = "dataset"

X = []
y = []

def extract_features(file_path):

    y_audio, sr = librosa.load(file_path, duration=30)

    # MFCC features
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y_audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral contrast
    spec_contrast = librosa.feature.spectral_contrast(y=y_audio, sr=sr)
    spec_contrast_mean = np.mean(spec_contrast, axis=1)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y_audio)
    zcr_mean = np.mean(zcr)

    # Combine all features
    features = np.concatenate([
        mfcc_mean,
        chroma_mean,
        spec_contrast_mean,
        [zcr_mean]
    ])

    return features


print("Extracting features...")

for genre in os.listdir(DATASET_PATH):

    if genre.startswith("."):
        continue

    genre_folder = os.path.join(DATASET_PATH, genre)

    for file in os.listdir(genre_folder):

        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(genre_folder, file)

        try:

            features = extract_features(file_path)

            X.append(features)
            y.append(genre)

        except Exception as e:
            print("Error processing:", file_path)


X = np.array(X)

print("Encoding labels...")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("Scaling features...")

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

print("Evaluating model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

print("Saving model...")

joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(label_encoder, "label.joblib")

print("Model trained and saved successfully!")
