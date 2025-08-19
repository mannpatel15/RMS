# # Implementation/src/train_models.py
# import os
# import joblib
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
# from tensorflow.keras.optimizers import Adam
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import OneClassSVM
# from sklearn.ensemble import IsolationForest
# from dotenv import load_dotenv

# load_dotenv()
# base_dir = os.getenv("base_dir")
# DATA = os.path.join(base_dir, "data", "cd_vector_clean.csv")
# MODEL_DIR = os.path.join(base_dir, "models")
# TIMESTEPS = 10  # keep consistent across train/predict

# os.makedirs(MODEL_DIR, exist_ok=True)

# def build_lstm_autoencoder(timesteps, n_features):
#     model = Sequential([
#         LSTM(64, activation="relu", input_shape=(timesteps, n_features), return_sequences=False),
#         RepeatVector(timesteps),
#         LSTM(64, activation="relu", return_sequences=True),
#         TimeDistributed(Dense(n_features))
#     ])
#     model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
#     return model

# def train():
#     df = pd.read_csv(DATA)

#     # Separate out metadata
#     for col in ["Label", "Timestamp"]:
#         if col in df.columns:
#             df = df.drop(columns=[col])

#     # Ensure numeric & finite
#     df = df.select_dtypes(include="number")
#     df = df.replace([np.inf, -np.inf], np.nan).dropna()

#     if df.empty:
#         raise ValueError("No numeric data to train on after cleaning.")

#     # Save feature order to enforce same order at prediction time
#     feature_names = df.columns.tolist()
#     joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

#     # Scale (fit on OWNER data only)
#     scaler = StandardScaler()
#     X = scaler.fit_transform(df.values)
#     joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

#     # ---- One-class models ----
#     ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
#     ocsvm.fit(X)
#     joblib.dump(ocsvm, os.path.join(MODEL_DIR, "oneclass_svm.pkl"))

#     iso = IsolationForest(contamination=0.05, random_state=42)
#     iso.fit(X)
#     joblib.dump(iso, os.path.join(MODEL_DIR, "isolation_forest.pkl"))

#     # ---- LSTM-AE ----
#     if len(X) <= TIMESTEPS:
#         raise ValueError(f"Not enough rows ({len(X)}) for timesteps={TIMESTEPS}")

#     X_seq = np.array([X[i:i+TIMESTEPS] for i in range(len(X) - TIMESTEPS)])
#     n_features = X.shape[1]

#     ae = build_lstm_autoencoder(TIMESTEPS, n_features)
#     ae.fit(X_seq, X_seq, epochs=20, batch_size=64, validation_split=0.1, shuffle=True, verbose=1)

#     # Calibrate threshold on train recon error
#     X_pred = ae.predict(X_seq, verbose=0)
#     train_err = np.mean((X_seq - X_pred) ** 2, axis=(1, 2))
#     thr = float(np.percentile(train_err, 95))  # tuneable

#     ae.save(os.path.join(MODEL_DIR, "lstm_autoencoder.h5"))
#     joblib.dump(thr, os.path.join(MODEL_DIR, "lstm_threshold.pkl"))

#     print(f"✅ Trained OCSVM, IF, LSTM-AE | features={len(feature_names)} | threshold={thr:.6f}")
#     print(f"📦 Saved to: {MODEL_DIR}")

# if __name__ == "__main__":
#     train()

# Implementation/src/train_models.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv("base_dir")
DATA = os.path.join(base_dir, "data", "cd_vector_clean.csv")
MODEL_DIR = os.path.join(base_dir, "models")
TIMESTEPS = 10  # keep consistent across train/predict

os.makedirs(MODEL_DIR, exist_ok=True)


# ----------------------------
# Build LSTM Autoencoder Model
# ----------------------------
def build_lstm_autoencoder(timesteps, n_features):
    model = Sequential([
        LSTM(64, activation="relu", input_shape=(timesteps, n_features), return_sequences=False),
        RepeatVector(timesteps),
        LSTM(64, activation="relu", return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return model


# ----------------------------
# Train all models
# ----------------------------
def train():
    df = pd.read_csv(DATA)

    # Drop metadata columns if present
    for col in ["Label", "Timestamp"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Keep only numeric
    df = df.select_dtypes(include="number")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if df.empty:
        raise ValueError("No numeric data to train on after cleaning.")

    # Save feature order to JSON
    feature_names = df.columns.tolist()
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Fit scaler
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # ---- One-class Models ----
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(X)
    joblib.dump(ocsvm, os.path.join(MODEL_DIR, "oneclass_svm.pkl"))

    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)
    joblib.dump(iso, os.path.join(MODEL_DIR, "isolation_forest.pkl"))

    # ---- LSTM Autoencoder ----
    if len(X) <= TIMESTEPS:
        raise ValueError(f"Not enough rows ({len(X)}) for timesteps={TIMESTEPS}")

    # Convert to sequences
    X_seq = np.array([X[i:i+TIMESTEPS] for i in range(len(X) - TIMESTEPS)])
    n_features = X.shape[1]

    ae = build_lstm_autoencoder(TIMESTEPS, n_features)
    ae.fit(X_seq, X_seq, epochs=20, batch_size=64, validation_split=0.1, shuffle=True, verbose=1)

    # Reconstruction error → threshold
    X_pred = ae.predict(X_seq, verbose=0)
    train_err = np.mean((X_seq - X_pred) ** 2, axis=(1, 2))
    threshold = float(np.percentile(train_err, 95))  # high-sensitivity

    # Save model + threshold
    ae.save(os.path.join(MODEL_DIR, "lstm_autoencoder.h5"))
    joblib.dump(threshold, os.path.join(MODEL_DIR, "lstm_threshold.pkl"))

    print(f"✅ Trained OCSVM, IF, LSTM-AE | features={len(feature_names)} | threshold={threshold:.6f}")
    print(f"📦 Saved all models to: {MODEL_DIR}")


if __name__ == "__main__":
    train()