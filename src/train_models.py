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


# #!/usr/bin/env python3
# """
# train_models.py

# Hybrid training (unsupervised + supervised RF if labels present).
# Saves:
#  - scaler.pkl, imputer.pkl, feature_names.json
#  - oneclass_svm.pkl, isolation_forest.pkl, lof.pkl, elliptic_envelope.pkl
#  - lstm_autoencoder.keras, lstm_threshold.pkl
#  - rcm_params.pkl
#  - rf_model.pkl (Random Forest, uses labels if available, else pseudo-labeling)
#  - training_meta.json (best_epoch, best_val_loss, etc.)
#  - owner_buffer.npy (last WINDOW_SIZE scaled owner rows for warm restarts)
# """
# import os
# import json
# import argparse
# import joblib
# import numpy as np
# import pandas as pd
# from dotenv import load_dotenv
# from pathlib import Path

# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import OneClassSVM
# from sklearn.ensemble import IsolationForest, RandomForestClassifier
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.covariance import EllipticEnvelope

# # tensorflow
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # reproducibility
# RND = 42
# np.random.seed(RND)
# tf.random.set_seed(RND)

# # Defaults
# WINDOW_SIZE = 5000  # rows to keep for warm restarts

# def ensure_dir(p):
#     os.makedirs(p, exist_ok=True)

# def sliding_window_sequences(X, timesteps):
#     n_rows, n_features = X.shape
#     if n_rows < timesteps:
#         return None
#     n_seq = n_rows - timesteps + 1
#     seqs = np.zeros((n_seq, timesteps, n_features), dtype=np.float32)
#     for i in range(n_seq):
#         seqs[i] = X[i:i+timesteps]
#     return seqs

# def build_lstm_autoencoder(timesteps, n_features, latent_dim=32):
#     inp = Input(shape=(timesteps, n_features))
#     x = LSTM(latent_dim, activation='tanh', return_sequences=False)(inp)
#     x = RepeatVector(timesteps)(x)
#     x = LSTM(latent_dim, activation='tanh', return_sequences=True)(x)
#     out = TimeDistributed(Dense(n_features))(x)
#     model = Model(inputs=inp, outputs=out)
#     model.compile(optimizer='adam', loss='mse')
#     return model

# def main(argv=None):
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--base_dir", type=str, default=None, help="Project base dir (overrides .env)")
#     parser.add_argument("--timesteps", type=int, default=10, help="LSTM timesteps")
#     parser.add_argument("--lstm_latent", type=int, default=32)
#     parser.add_argument("--lstm_epochs", type=int, default=50)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--val_split", type=float, default=0.1)
#     parser.add_argument("--lstm_std_mult", type=float, default=2.0)
#     args = parser.parse_args(argv)

#     load_dotenv()
#     base_dir = args.base_dir or os.getenv("base_dir") or str(Path(__file__).resolve().parent.parent)
#     base_dir = str(Path(base_dir).resolve())
#     print("[INFO] base_dir =", base_dir)

#     data_path = os.path.join(base_dir, "data", "cd_vector.csv")
#     model_dir = os.path.join(base_dir, "models")
#     ensure_dir(model_dir)

#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"cd_vector.csv not found at {data_path}")

#     print("[INFO] Loading csv:", data_path)
#     df = pd.read_csv(data_path)
#     print("[INFO] Raw columns:", list(df.columns))

#     # check label column
#     label_col = None
#     for lbl in ["Label", "label"]:
#         if lbl in df.columns:
#             label_col = lbl
#             break

#     if label_col:
#         print(f"[INFO] Found supervised label column '{label_col}'")
#         y = df[label_col].values
#         df = df.drop(columns=[label_col])
#     else:
#         print("[WARN] No label column found → RF will use pseudo-labeling")
#         y = None

#     # drop timestamps if exist
#     for tcol in ["Timestamp", "timestamp", "ts"]:
#         if tcol in df.columns:
#             df = df.drop(columns=[tcol])

#     # keep numeric
#     df_num = df.select_dtypes(include=[np.number]).copy()
#     numeric_cols = df_num.columns.tolist()
#     if len(numeric_cols) == 0:
#         raise RuntimeError("No numeric features found in cd_vector.csv")

#     # imputation
#     count_cols = [c for c in numeric_cols if "count" in c.lower()]
#     imputer_map = {}
#     for c in numeric_cols:
#         if c in count_cols:
#             imputer_map[c] = 0.0
#             df_num[c] = df_num[c].fillna(0.0)
#         else:
#             med = df_num[c].median()
#             if np.isnan(med):
#                 med = 0.0
#             imputer_map[c] = float(med)
#             df_num[c] = df_num[c].fillna(med)

#     joblib.dump({"imputer": imputer_map, "count_cols": count_cols}, os.path.join(model_dir, "imputer.pkl"))
#     with open(os.path.join(model_dir, "feature_names.json"), "w") as fh:
#         json.dump(numeric_cols, fh)

#     X = df_num.values.astype(np.float32)
#     print(f"[INFO] Final dataset shape: {X.shape}")

#     # scaling
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

#     # ------------------ Classical unsupervised detectors ------------------
#     print("[INFO] Training OneClassSVM...")
#     ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_scaled)
#     joblib.dump(ocsvm, os.path.join(model_dir, "oneclass_svm.pkl"))

#     print("[INFO] Training IsolationForest...")
#     iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=RND, n_jobs=-1).fit(X_scaled)
#     joblib.dump(iso, os.path.join(model_dir, "isolation_forest.pkl"))

#     print("[INFO] Training LocalOutlierFactor...")
#     lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.01).fit(X_scaled)
#     joblib.dump(lof, os.path.join(model_dir, "lof.pkl"))

#     print("[INFO] Training EllipticEnvelope...")
#     try:
#         ee = EllipticEnvelope(random_state=RND, contamination=0.01).fit(X_scaled)
#         joblib.dump(ee, os.path.join(model_dir, "elliptic_envelope.pkl"))
#     except Exception as e:
#         print("[WARN] EllipticEnvelope failed:", e)

#     # ------------------ LSTM Autoencoder ------------------
#     timesteps = args.timesteps
#     seqs = sliding_window_sequences(X_scaled, timesteps)
#     training_meta = {"lstm_trained": False}

#     if seqs is not None:
#         n_features = seqs.shape[2]
#         ae = build_lstm_autoencoder(timesteps, n_features, args.lstm_latent)
#         keras_path = os.path.join(model_dir, "lstm_autoencoder.keras")
#         callbacks = [
#             ModelCheckpoint(keras_path, save_best_only=True, monitor="val_loss", mode="min"),
#             EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
#         ]

#         n_seq = seqs.shape[0]
#         val_n = max(int(n_seq * args.val_split), min(10, int(0.05 * n_seq)))
#         train_seqs = seqs[:-val_n] if val_n > 0 else seqs
#         val_seqs = seqs[-val_n:] if val_n > 0 else seqs[:0]

#         history = ae.fit(
#             train_seqs, train_seqs,
#             epochs=args.lstm_epochs,
#             batch_size=args.batch_size,
#             validation_data=(val_seqs, val_seqs) if len(val_seqs) else None,
#             callbacks=callbacks,
#             verbose=1
#         )

#         ae.save(keras_path)
#         print("[INFO] LSTM Autoencoder saved")

#         seq_pred = ae.predict(seqs, batch_size=args.batch_size)
#         seq_err = np.mean((seqs - seq_pred) ** 2, axis=(1, 2))
#         thr = float(np.median(seq_err) + args.lstm_std_mult * np.std(seq_err))
#         joblib.dump(thr, os.path.join(model_dir, "lstm_threshold.pkl"))

#         training_meta.update({
#             "lstm_trained": True,
#             "best_epoch": int(np.argmin(history.history["val_loss"]) + 1) if "val_loss" in history.history else None,
#             "best_val_loss": float(np.min(history.history["val_loss"])) if "val_loss" in history.history else None
#         })

#     # ------------------ RCM params ------------------
#     errs = np.mean((X_scaled - np.mean(X_scaled, axis=0)) ** 2, axis=1)
#     training_meta["median_err"] = float(np.median(errs))
#     training_meta["std_err"] = float(np.std(errs))
#     rcm = {"median": training_meta["median_err"], "std": training_meta["std_err"], "alpha": 0.05}
#     joblib.dump(rcm, os.path.join(model_dir, "rcm_params.pkl"))

#     # ------------------ Random Forest (supervised if possible) ------------------
#     print("[INFO] Training RandomForest...")
#     if y is not None:
#         rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RND, n_jobs=-1)
#         rf.fit(X_scaled, y)
#     else:
#         # pseudo-label: all owners=1, synthetic Gaussian noise=0
#         noise = np.random.normal(0, 1, size=X_scaled.shape)
#         X_aug = np.vstack([X_scaled, noise])
#         y_aug = np.hstack([np.ones(X_scaled.shape[0]), np.zeros(noise.shape[0])])
#         rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RND, n_jobs=-1)
#         rf.fit(X_aug, y_aug)

#     joblib.dump(rf, os.path.join(model_dir, "rf_model.pkl"))
#     print("[INFO] rf_model.pkl saved")

#     # ------------------ Save buffers & meta ------------------
#     last_n = min(WINDOW_SIZE, X_scaled.shape[0])
#     np.save(os.path.join(model_dir, "owner_buffer.npy"), X_scaled[-last_n:, :].astype(np.float32))

#     with open(os.path.join(model_dir, "training_meta.json"), "w") as fh:
#         json.dump(training_meta, fh, indent=2)

#     print("\n=== ARTIFACTS SAVED IN", model_dir, "===")
#     for f in sorted(os.listdir(model_dir)):
#         print(" -", f)
#     print("=== training complete ===")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
train_models.py

Trains a suite of unsupervised and supervised models for CUA, then creates
the initial state file required by the online monitor.
"""
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Reproducibility ---
RND_STATE = 42
np.random.seed(RND_STATE)
tf.random.set_seed(RND_STATE)

def sliding_window_sequences(X, timesteps):
    """Converts a 2D array of features into 3D sequences for LSTM."""
    n_rows = X.shape[0]
    if n_rows < timesteps:
        return None
    # Using a more efficient list comprehension method
    return np.array([X[i:(i + timesteps)] for i in range(n_rows - timesteps + 1)])

def build_lstm_autoencoder(timesteps, n_features, latent_dim=32):
    """Builds the LSTM Autoencoder model architecture."""
    inp = Input(shape=(timesteps, n_features))
    # Encoder
    x = LSTM(latent_dim, activation='tanh', return_sequences=False)(inp)
    # Decoder
    x = RepeatVector(timesteps)(x)
    x = LSTM(latent_dim, activation='tanh', return_sequences=True)(x)
    out = TimeDistributed(Dense(n_features))(x)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

def main(argv=None):
    parser = argparse.ArgumentParser(description="CUA Model Training Script")
    parser.add_argument("--base_dir", type=str, default=None, help="Project base directory (overrides .env)")
    parser.add_argument("--timesteps", type=int, default=10, help="Sequence length for LSTM model")
    parser.add_argument("--lstm_epochs", type=int, default=50, help="Max epochs for LSTM training")
    args = parser.parse_args(argv)

    # --- 1. Setup Paths ---
    load_dotenv()
    base_dir = Path(args.base_dir or os.getenv("base_dir") or Path(__file__).resolve().parent.parent)
    data_path = base_dir / "data" / "cd_vector.csv"
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    print(f"[INFO] Using base directory: {base_dir}")

    # --- 2. Load and Preprocess Data ---
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    print(f"[INFO] Loading data from: {data_path}")
    # Fine-tuning: Added on_bad_lines='warn' to gracefully handle corrupted CSV rows.
    df = pd.read_csv(data_path, on_bad_lines='warn')

    # Drop non-feature columns
    df = df.drop(columns=["Label", "Timestamp"], errors='ignore')
    
    df_num = df.select_dtypes(include=np.number).copy()
    numeric_cols = df_num.columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric features found in the data file.")

    # Impute missing values (fill NaNs)
    imputer_map = {}
    for col in numeric_cols:
        fill_value = 0.0 if "count" in col.lower() else df_num[col].median()
        if np.isnan(fill_value): fill_value = 0.0
        # ### THIS IS THE FIX ###
        # This is the modern, safe way to fill missing values
        df_num[col] = df_num[col].fillna(fill_value) 
        imputer_map[col] = float(fill_value)

    X = df_num.values.astype(np.float32)
    print(f"[INFO] Final dataset shape for training: {X.shape}")

    # --- 3. Save Preprocessing Artifacts ---
    joblib.dump({"imputer": imputer_map}, model_dir / "imputer.pkl")
    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(numeric_cols, f)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, model_dir / "scaler.pkl")

    # --- 4. Train Unsupervised Models ---
    print("\n[INFO] Training Unsupervised Anomaly Detectors...")
    # These contamination/nu values assume the training data is mostly from the owner.
    joblib.dump(OneClassSVM(nu=0.05).fit(X_scaled), model_dir / "oneclass_svm.pkl")
    joblib.dump(IsolationForest(contamination=0.01, random_state=RND_STATE).fit(X_scaled), model_dir / "isolation_forest.pkl")
    joblib.dump(LocalOutlierFactor(novelty=True, contamination=0.01).fit(X_scaled), model_dir / "lof.pkl")
    try:
        joblib.dump(EllipticEnvelope(contamination=0.01, random_state=RND_STATE).fit(X_scaled), model_dir / "elliptic_envelope.pkl")
    except ValueError as e:
        print(f"[WARN] EllipticEnvelope failed. Skipping. Error: {e}")
    print("✅ Unsupervised models trained.")

    # --- 5. Train LSTM Autoencoder ---
    print("\n[INFO] Training LSTM Autoencoder...")
    timesteps = args.timesteps
    seqs = sliding_window_sequences(X_scaled, timesteps)
    
    if seqs is not None:
        n_features = seqs.shape[2]
        ae = build_lstm_autoencoder(timesteps, n_features)
        keras_path = model_dir / "lstm_autoencoder.keras"
        
        callbacks = [
            ModelCheckpoint(str(keras_path), save_best_only=True, monitor="val_loss", mode="min"),
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        ]

        ae.fit(seqs, seqs, epochs=args.lstm_epochs, batch_size=64, validation_split=0.1, callbacks=callbacks, verbose=1)
        
        seq_pred = ae.predict(seqs, batch_size=64)
        seq_err = np.mean(np.square(seqs - seq_pred), axis=(1, 2))
        # This robust threshold works well for anomaly detection
        threshold = float(np.mean(seq_err) + 2 * np.std(seq_err))
        joblib.dump(threshold, model_dir / "lstm_threshold.pkl")
        print(f"✅ LSTM Autoencoder trained. Anomaly threshold set to: {threshold:.6f}")
    else:
        print(f"[WARN] Not enough data ({X.shape[0]} rows) to train LSTM with {timesteps} timesteps. Skipping.")

    # --- 6. Train RandomForest Classifier (with Pseudo-Labeling) ---
    print("\n[INFO] Training RandomForest with Pseudo-Labeling...")
    noise = np.random.normal(0, 1, size=X_scaled.shape)
    X_aug = np.vstack([X_scaled, noise])
    y_aug = np.hstack([np.ones(X_scaled.shape[0]), np.zeros(noise.shape[0])])
    
    # Fine-tuning: n_estimators=100 is a great balance of speed and performance.
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RND_STATE, n_jobs=-1)
    rf.fit(X_aug, y_aug)
    joblib.dump(rf, model_dir / "rf_model.pkl")
    print("✅ RandomForest trained.")

    # --- 7. CRITICAL STEP: Create the Initial State File for the Monitor ---
    print("\n[INFO] Creating initial state file for the monitor...")
    initial_state = {
        "last_processed_row": X.shape[0],
        "model_version": 1,
        "confidence_score": 1.0,
        "trusted_buffer_size": 0
    }
    with open(base_dir / "cua_state.json", "w") as f:
        json.dump(initial_state, f, indent=4)
    print(f"✅ 'cua_state.json' created. Monitor will now start processing from row {X.shape[0] + 1}.")
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()