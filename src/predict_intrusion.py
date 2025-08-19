# Implementation/src/predict_intrusion.py

import os
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv("base_dir")
MODEL_DIR = os.path.join(base_dir, "models")

def predict_intrusion(test_file, threshold=0.8):
    # Load test data
    df = pd.read_csv(test_file)

    # Drop non-numeric columns
    for col in ["Label", "Timestamp"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Drop empty columns
    df = df.dropna(axis=1, how="all")

    # Fill NaN with column mean
    df = df.fillna(df.mean(numeric_only=True))

    # Load scaler and transform
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    X_scaled = scaler.transform(df.values)

    # Reshape for LSTM [samples, timesteps, features]
    timesteps = 10
    n_features = X_scaled.shape[1]
    X_seq = []
    for i in range(len(X_scaled) - timesteps):
        X_seq.append(X_scaled[i:i+timesteps])
    X_seq = np.array(X_seq)

    # Load trained autoencoder
    model = load_model(os.path.join(MODEL_DIR, "lstm_autoencoder.h5"))

    # Reconstruct and compute errors
    X_pred = model.predict(X_seq)
    errors = np.mean(np.power(X_seq - X_pred, 2), axis=(1,2))

    # Intrusion decision
    predictions = (errors > threshold).astype(int)  # 0=owner, 1=intruder
    results = pd.DataFrame({
        "ReconstructionError": errors,
        "Prediction": predictions
    })

    return results

if __name__ == "__main__":
    test_file = os.path.join(base_dir, "data", "cd_vector.csv")  # test from recent data
    results = predict_intrusion(test_file)
    print(results.head())