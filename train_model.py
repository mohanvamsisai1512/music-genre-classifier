import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# =========================
# LOAD DATASET (CSV)
# =========================

df = pd.read_csv("features_30_sec.csv")   # place this file in your project

# =========================
# SELECT MFCC FEATURES
# =========================

mfcc_columns = [col for col in df.columns if 'mfcc' in col]

X = df[mfcc_columns]
y = df['label']

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# SCALING
# =========================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# TRAIN MODEL (MLP)
# =========================

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model trained")
print("Accuracy:", round(accuracy * 100, 2), "%")

# =========================
# SAVE FILES
# =========================

joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(encoder, "label.joblib")
joblib.dump(mfcc_columns, "columns.joblib")

print("✅ Model saved")