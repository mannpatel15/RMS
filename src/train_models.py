# Implementation/src/train_models.py
import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv('base_dir')
DATA_PATH = os.path.join(base_dir, "data", "cd_vector.csv")
MODEL_DIR = os.path.join(base_dir, "models")

def build_lstm_autoencoder(timesteps, n_features):
    model = Sequential([
        LSTM(64, activation="relu", input_shape=(timesteps, n_features), return_sequences=False),
        RepeatVector(timesteps),
        LSTM(64, activation="relu", return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

def train_lstm_autoencoder():
    df = pd.read_csv(DATA_PATH)

    # Drop non-numeric cols
    for col in ["Label", "Timestamp"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Clean
    df = df.dropna(axis=1, how="all")
    df = df.fillna(df.mean(numeric_only=True))
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    # Reshape for sequences
    timesteps = 10
    n_features = X_scaled.shape[1]
    X_seq = np.array([X_scaled[i:i+timesteps] for i in range(len(X_scaled) - timesteps)])

    # Build + train
    model = build_lstm_autoencoder(timesteps, n_features)
    model.fit(X_seq, X_seq, epochs=20, batch_size=32, validation_split=0.1, shuffle=True)

    # Reconstruction errors on training set
    X_pred_train = model.predict(X_seq, verbose=0)
    train_errors = np.mean(np.power(X_seq - X_pred_train, 2), axis=(1, 2))
    threshold = float(np.percentile(train_errors, 95))

    # Save everything
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "lstm_autoencoder.h5"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(threshold, os.path.join(MODEL_DIR, "lstm_threshold.pkl"))

    print(f"✅ LSTM Autoencoder trained and saved. Threshold: {threshold:.6f}")

if __name__ == "__main__":
    train_lstm_autoencoder()