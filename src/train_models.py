# # # Implementation/src/train_models.py
# # import os
# # import joblib
# # import numpy as np
# # import pandas as pd
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
# # from tensorflow.keras.optimizers import Adam
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.svm import OneClassSVM
# # from sklearn.ensemble import IsolationForest
# from dotenv import load_dotenv

# load_dotenv()
# base_dir = os.getenv("base_dir")
# # DATA = os.path.join(base_dir, "data", "cd_vector_clean.csv")
# # MODEL_DIR = os.path.join(base_dir, "models")
# # TIMESTEPS = 10  # keep consistent across train/predict

# # os.makedirs(MODEL_DIR, exist_ok=True)

# # def build_lstm_autoencoder(timesteps, n_features):
# #     model = Sequential([
# #         LSTM(64, activation="relu", input_shape=(timesteps, n_features), return_sequences=False),
# #         RepeatVector(timesteps),
# #         LSTM(64, activation="relu", return_sequences=True),
# #         TimeDistributed(Dense(n_features))
# #     ])
# #     model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
# #     return model

# # def train():
# #     df = pd.read_csv(DATA)

# #     # Separate out metadata
# #     for col in ["Label", "Timestamp"]:
# #         if col in df.columns:
# #             df = df.drop(columns=[col])

# #     # Ensure numeric & finite
# #     df = df.select_dtypes(include="number")
# #     df = df.replace([np.inf, -np.inf], np.nan).dropna()

# #     if df.empty:
# #         raise ValueError("No numeric data to train on after cleaning.")

# #     # Save feature order to enforce same order at prediction time
# #     feature_names = df.columns.tolist()
# #     joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

# #     # Scale (fit on OWNER data only)
# #     scaler = StandardScaler()
# #     X = scaler.fit_transform(df.values)
# #     joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# #     # ---- One-class models ----
# #     ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
# #     ocsvm.fit(X)
# #     joblib.dump(ocsvm, os.path.join(MODEL_DIR, "oneclass_svm.pkl"))

# #     iso = IsolationForest(contamination=0.05, random_state=42)
# #     iso.fit(X)
# #     joblib.dump(iso, os.path.join(MODEL_DIR, "isolation_forest.pkl"))

# #     # ---- LSTM-AE ----
# #     if len(X) <= TIMESTEPS:
# #         raise ValueError(f"Not enough rows ({len(X)}) for timesteps={TIMESTEPS}")

# #     X_seq = np.array([X[i:i+TIMESTEPS] for i in range(len(X) - TIMESTEPS)])
# #     n_features = X.shape[1]

# #     ae = build_lstm_autoencoder(TIMESTEPS, n_features)
# #     ae.fit(X_seq, X_seq, epochs=20, batch_size=64, validation_split=0.1, shuffle=True, verbose=1)

# #     # Calibrate threshold on train recon error
# #     X_pred = ae.predict(X_seq, verbose=0)
# #     train_err = np.mean((X_seq - X_pred) ** 2, axis=(1, 2))
# #     thr = float(np.percentile(train_err, 95))  # tuneable

# #     ae.save(os.path.join(MODEL_DIR, "lstm_autoencoder.h5"))
# #     joblib.dump(thr, os.path.join(MODEL_DIR, "lstm_threshold.pkl"))

# #     print(f"✅ Trained OCSVM, IF, LSTM-AE | features={len(feature_names)} | threshold={thr:.6f}")
# #     print(f"📦 Saved to: {MODEL_DIR}")

# # if __name__ == "__main__":
# #     train()

# # Implementation/src/train_models.py
# import os
# import json
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


# # ----------------------------
# # Build LSTM Autoencoder Model
# # ----------------------------
# def build_lstm_autoencoder(timesteps, n_features):
#     model = Sequential([
#         LSTM(64, activation="relu", input_shape=(timesteps, n_features), return_sequences=False),
#         RepeatVector(timesteps),
#         LSTM(64, activation="relu", return_sequences=True),
#         TimeDistributed(Dense(n_features))
#     ])
#     model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
#     return model


# # ----------------------------
# # Train all models
# # ----------------------------
# def train():
#     df = pd.read_csv(DATA)

#     # Drop metadata columns if present
#     for col in ["Label", "Timestamp"]:
#         if col in df.columns:
#             df = df.drop(columns=[col])

#     # Keep only numeric
#     df = df.select_dtypes(include="number")
#     df = df.replace([np.inf, -np.inf], np.nan).dropna()

#     if df.empty:
#         raise ValueError("No numeric data to train on after cleaning.")

#     # Save feature order to JSON
#     feature_names = df.columns.tolist()
#     with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
#         json.dump(feature_names, f)

#     # Fit scaler
#     scaler = StandardScaler()
#     X = scaler.fit_transform(df.values)
#     joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

#     # ---- One-class Models ----
#     ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
#     ocsvm.fit(X)
#     joblib.dump(ocsvm, os.path.join(MODEL_DIR, "oneclass_svm.pkl"))

#     iso = IsolationForest(contamination=0.05, random_state=42)
#     iso.fit(X)
#     joblib.dump(iso, os.path.join(MODEL_DIR, "isolation_forest.pkl"))

#     # ---- LSTM Autoencoder ----
#     if len(X) <= TIMESTEPS:
#         raise ValueError(f"Not enough rows ({len(X)}) for timesteps={TIMESTEPS}")

#     # Convert to sequences
#     X_seq = np.array([X[i:i+TIMESTEPS] for i in range(len(X) - TIMESTEPS)])
#     n_features = X.shape[1]

#     ae = build_lstm_autoencoder(TIMESTEPS, n_features)
#     ae.fit(X_seq, X_seq, epochs=20, batch_size=64, validation_split=0.1, shuffle=True, verbose=1)

#     # Reconstruction error → threshold
#     X_pred = ae.predict(X_seq, verbose=0)
#     train_err = np.mean((X_seq - X_pred) ** 2, axis=(1, 2))
#     threshold = float(np.percentile(train_err, 95))  # high-sensitivity

#     # Save model + threshold
#     ae.save(os.path.join(MODEL_DIR, "lstm_autoencoder.h5"))
#     joblib.dump(threshold, os.path.join(MODEL_DIR, "lstm_threshold.pkl"))

#     print(f"✅ Trained OCSVM, IF, LSTM-AE | features={len(feature_names)} | threshold={threshold:.6f}")
#     print(f"📦 Saved all models to: {MODEL_DIR}")


# if __name__ == "__main__":
#     train()


#!/usr/bin/env python3
"""
train_models.py

Unsupervised training on owner-only cd_vectors.

Saves:
 - models/: oc_svm, isolation_forest, lof (novelty), lstm_ae.h5
 - lstm_threshold.pkl
 - scaler.pkl, imputer.pkl, feature_names.json
 - rcm_params.pkl
 - ensemble_config.json

Usage:
  python src/train_models.py
  python src/train_models.py --base_dir /path/to/project --timesteps 10 --lstm_epochs 20
"""
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# reproducibility
RND = 42
np.random.seed(RND)
tf.random.set_seed(RND)

# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def sliding_window_sequences(X, timesteps):
    """Make overlapping sequences of shape (n_seq, timesteps, n_features)."""
    n_rows, n_features = X.shape
    if n_rows < timesteps:
        return None
    n_seq = n_rows - timesteps + 1
    seqs = np.zeros((n_seq, timesteps, n_features), dtype=np.float32)
    for i in range(n_seq):
        seqs[i] = X[i:i+timesteps]
    return seqs

def build_lstm_autoencoder(timesteps, n_features, latent_dim=32):
    inp = Input(shape=(timesteps, n_features))
    x = LSTM(latent_dim, activation='tanh', return_sequences=False)(inp)
    x = RepeatVector(timesteps)(x)
    x = LSTM(latent_dim, activation='tanh', return_sequences=True)(x)
    out = TimeDistributed(Dense(n_features))(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------------------
# RCM: simple parametric mechanism
#   stores baseline median/std and smoothing alpha
#   score(err) returns confidence in [0,1] (higher -> more owner-like)
# ---------------------------
class RCM:
    def __init__(self, median=0.0, std=1.0, alpha=0.1):
        self.median = float(median)
        self.std = float(std) if std > 0 else 1.0
        self.alpha = float(alpha)
        self.moving_confidence = 1.0

    def score(self, err):
        # normalized anomaly score: z = (err - median) / std
        z = (err - self.median) / (self.std + 1e-9)
        # map to [0,1] with soft clamp: owner-like -> near 1, anomalous -> near 0
        conf = 1.0 - (1.0 / (1.0 + np.exp(- ( -z + 1.0 ))))  # sigmoid-ish invert
        # simpler linear fallback:
        conf_lin = max(0.0, 1.0 - min(3.0, z) / 3.0)
        # combine
        conf = 0.6 * conf + 0.4 * conf_lin
        # update moving confidence (exponential smoothing)
        self.moving_confidence = (1 - self.alpha) * self.moving_confidence + self.alpha * conf
        return float(conf)

    def to_dict(self):
        return {"median": self.median, "std": self.std, "alpha": self.alpha, "moving_confidence": self.moving_confidence}

# ---------------------------
# Main training flow
# ---------------------------
def main(args):
    # load base_dir from env if not passed
    load_dotenv()
    base_dir = args.base_dir or os.getenv("base_dir") or str(Path(__file__).resolve().parent.parent)
    base_dir = str(Path(base_dir).resolve())
    print("[INFO] base_dir =", base_dir)

    data_path = os.path.join(base_dir, "data", "cd_vector.csv")
    model_dir = os.path.join(base_dir, "models")
    ensure_dir(model_dir)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"cd_vector.csv not found at {data_path}")

    print("[INFO] Loading csv:", data_path)
    df = pd.read_csv(data_path)

    # Save columns for debugging
    print("[INFO] Raw columns:", list(df.columns))

    # Remove Label column if present (we're unsupervised)
    for lbl in ["Label", "label", "target", "class"]:
        if lbl in df.columns:
            print(f"[INFO] Dropping column '{lbl}' (unsupervised training)")
            df = df.drop(columns=[lbl])

    # Keep only numeric columns (drop timestamps / object)
    df_num = df.select_dtypes(include=[np.number]).copy()
    if df_num.shape[1] == 0:
        raise RuntimeError("No numeric columns found after selecting dtypes. Inspect cd_vector.csv")

    print("[INFO] Numeric features kept:", df_num.shape[1])

    # Imputation strategy: count-like -> 0, others -> median
    numeric_cols = df_num.columns.tolist()
    count_cols = [c for c in numeric_cols if 'count' in c.lower()]
    imputer = {}
    for c in numeric_cols:
        if c in count_cols:
            imputer[c] = 0.0
            df_num[c] = df_num[c].fillna(0.0)
        else:
            med = df_num[c].median()
            if np.isnan(med):
                med = 0.0
            imputer[c] = float(med)
            df_num[c] = df_num[c].fillna(med)

    # Save imputer and feature order
    joblib.dump({"imputer": imputer, "count_cols": count_cols}, os.path.join(model_dir, "imputer.pkl"))
    with open(os.path.join(model_dir, "feature_names.json"), "w") as fh:
        json.dump(numeric_cols, fh)

    # Convert to numpy floats
    X = df_num.values.astype(np.float32)
    print(f"[INFO] After imputation dataset shape: {X.shape}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print("[INFO] Scaler saved")

    # ---------------------------
    # Classical unsupervised detectors
    # ---------------------------
    print("[INFO] Training OneClassSVM...")
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=args.ocsvm_nu)
    ocsvm.fit(X_scaled)
    joblib.dump(ocsvm, os.path.join(model_dir, "oneclass_svm.pkl"))

    print("[INFO] Training IsolationForest...")
    iso = IsolationForest(n_estimators=200, contamination=args.iso_contamination, random_state=RND, n_jobs=-1)
    iso.fit(X_scaled)
    joblib.dump(iso, os.path.join(model_dir, "isolation_forest.pkl"))

    print("[INFO] Training LocalOutlierFactor (novelty=True)...")
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=args.lof_contamination)
    lof.fit(X_scaled)
    joblib.dump(lof, os.path.join(model_dir, "lof.pkl"))

    print("[INFO] Training EllipticEnvelope (Robust Covariance)...")
    try:
        ee = EllipticEnvelope(random_state=RND, support_fraction=None, contamination=args.ee_contamination)
        ee.fit(X_scaled)
        joblib.dump(ee, os.path.join(model_dir, "elliptic_envelope.pkl"))
    except Exception as e:
        print("[WARN] EllipticEnvelope training failed:", e)

    # Optional pseudo RandomForest (unsupervised hack) — disabled by default
    if args.use_pseudo_rf:
        print("[INFO] Creating pseudo-negative samples for RandomForest (debug only!)")
        n = X_scaled.shape[0]
        noise = X_scaled + np.random.normal(0, 0.5 * np.std(X_scaled, axis=0), size=X_scaled.shape)
        X_rf = np.vstack([X_scaled, noise])
        y_rf = np.hstack([np.ones(n), np.zeros(n)])
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=200, random_state=RND, n_jobs=-1)
        rf.fit(X_rf, y_rf)
        joblib.dump(rf, os.path.join(model_dir, "pseudo_rf.pkl"))
        print("[WARN] pseudo RandomForest saved (use only for debugging)")

    # ---------------------------
    # LSTM Autoencoder
    # ---------------------------
    timesteps = args.timesteps
    seqs = sliding_window_sequences(X_scaled, timesteps)

    if seqs is None:
        print(f"[WARN] Not enough rows ({X_scaled.shape[0]}) to build LSTM sequences (timesteps={timesteps}). Skipping LSTM AE.")
        lstm_trained = False
    else:
        lstm_trained = True
        print("[INFO] Training LSTM Autoencoder on sequences:", seqs.shape)
        n_features = seqs.shape[2]
        lstm = build_lstm_autoencoder(timesteps, n_features, latent_dim=args.lstm_latent)

        lstm_path = os.path.join(model_dir, "lstm_autoencoder.h5")
        checkpoint = ModelCheckpoint(lstm_path, save_best_only=True, monitor='val_loss', mode='min', verbose=0)
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)

        # simple train/val split
        n_seq = seqs.shape[0]
        val_n = max(int(n_seq * args.val_split), min(10, int(0.05 * n_seq)))
        train_seqs = seqs[:-val_n] if val_n > 0 else seqs
        val_seqs = seqs[-val_n:] if val_n > 0 else seqs[:0]

        lstm.fit(train_seqs, train_seqs,
                 epochs=args.lstm_epochs,
                 batch_size=args.batch_size,
                 validation_data=(val_seqs, val_seqs) if len(val_seqs) else None,
                 callbacks=[es, checkpoint],
                 verbose=1)

        # ensure best saved model exists
        lstm.save(lstm_path)
        print("[INFO] LSTM Autoencoder saved:", lstm_path)

        # Compute sequence reconstruction errors
        seq_pred = lstm.predict(seqs, batch_size=args.batch_size)
        seq_err = np.mean((seqs - seq_pred) ** 2, axis=(1, 2))  # per-sequence MSE
        thr = float(np.median(seq_err) + args.lstm_std_mult * np.std(seq_err))
        joblib.dump(thr, os.path.join(model_dir, "lstm_threshold.pkl"))
        print(f"[INFO] LSTM threshold saved (median + {args.lstm_std_mult}*std) = {thr:.6f}")

    # ---------------------------
    # RCM: derive parameters from LSTM errors (or fallback)
    # ---------------------------
    if lstm_trained:
        median_err = float(np.median(seq_err))
        std_err = float(np.std(seq_err))
    else:
        # fallback: compute simple reconstruction error from PCA-like distance
        errs = np.mean((X_scaled - np.mean(X_scaled, axis=0))**2, axis=1)
        median_err = float(np.median(errs))
        std_err = float(np.std(errs))

    rcm = RCM(median=median_err, std=std_err, alpha=args.rcm_alpha)
    joblib.dump(rcm.to_dict(), os.path.join(model_dir, "rcm_params.pkl"))
    print("[INFO] RCM params saved:", rcm.to_dict())

    # ---------------------------
    # Ensemble config (how the monitor should combine)
    # ---------------------------
    ensemble_cfg = {
        "detectors": ["oneclass_svm", "isolation_forest", "lof", "elliptic_envelope"],
        "lstm_autoencoder": lstm_trained,
        "lstm_threshold": "lstm_threshold.pkl" if lstm_trained else None,
        "voting_rule": {
            "vote_threshold": args.vote_threshold,   # how many detectors must signal anomaly to mark as anomaly
            "window_k": args.detect_k,
            "window_w": args.detect_w
        }
    }
    with open(os.path.join(model_dir, "ensemble_config.json"), "w") as fh:
        json.dump(ensemble_cfg, fh, indent=2)
    print("[INFO] Ensemble config saved")

    # Summary
    print("\n=== ARTIFACTS SAVED IN", model_dir, "===")
    for f in sorted(os.listdir(model_dir)):
        print(" -", f)
    print("=== training complete ===")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=None, help="Project base dir (overrides .env)")
    parser.add_argument("--timesteps", type=int, default=10, help="LSTM timesteps (sequence length)")
    parser.add_argument("--lstm_latent", type=int, default=32)
    parser.add_argument("--lstm_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--lstm_std_mult", type=float, default=2.0, help="threshold = median + mult * std")
    parser.add_argument("--ocsvm_nu", type=float, default=0.05)
    parser.add_argument("--iso_contamination", type=float, default=0.01)
    parser.add_argument("--lof_contamination", type=float, default=0.01)
    parser.add_argument("--ee_contamination", type=float, default=0.01)
    parser.add_argument("--rcm_alpha", type=float, default=0.05)
    parser.add_argument("--vote_threshold", type=int, default=2)
    parser.add_argument("--detect_k", type=int, default=4)
    parser.add_argument("--detect_w", type=int, default=6)
    parser.add_argument("--use_pseudo_rf", action="store_true", help="Train pseudo-RF by creating synthetic negatives (debug only)")
    args = parser.parse_args()
    main(args)