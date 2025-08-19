# Implementation/src/predict_intrusion.py
import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv("base_dir")
DATA = os.path.join(base_dir, "data", "cd_vector_clean.csv")
MODEL_DIR = os.path.join(base_dir, "models")
TIMESTEPS = 10

def predict_lstm(input_csv=DATA):
    raw = pd.read_csv(input_csv)

    # Load feature order & scaler
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    # Build feature matrix in the exact trained order
    X_df = raw.copy()
    for col in ["Label", "Timestamp"]:
        if col in X_df.columns:
            X_df = X_df.drop(columns=[col])
    X_df = X_df.reindex(columns=feature_names)  # missing -> NaN
    X_df = X_df.fillna(X_df.mean(numeric_only=True))
    X = scaler.transform(X_df.values)

    # Sequences
    if len(X) <= TIMESTEPS:
        raise ValueError(f"Need >{TIMESTEPS} rows, got {len(X)}")

    X_seq = np.array([X[i:i+TIMESTEPS] for i in range(len(X) - TIMESTEPS)])

    # Load model + threshold
    ae = load_model(os.path.join(MODEL_DIR, "lstm_autoencoder.h5"))
    thr = float(joblib.load(os.path.join(MODEL_DIR, "lstm_threshold.pkl")))

    X_pred = ae.predict(X_seq, verbose=0)
    seq_err = np.mean((X_seq - X_pred) ** 2, axis=(1, 2))

    # Align to rows (first TIMESTEPS rows have no seq)
    err_aligned = np.full(shape=(len(X),), fill_value=np.nan, dtype=float)
    err_aligned[TIMESTEPS:] = seq_err

    preds = (err_aligned > thr).astype(float)  # 1=intruder
    out = pd.DataFrame({
        "ReconstructionError": err_aligned,
        "Prediction": preds
    })
    return out

if __name__ == "__main__":
    res = predict_lstm()
    print(res.head())