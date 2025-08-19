# Implementation/src/monitor_realtime.py

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

load_dotenv()
base_dir = os.getenv("base_dir")
DATA_CSV   = os.path.join(base_dir, "data", "cd_vector.csv")
MODEL_DIR  = os.path.join(base_dir, "models")
POLL_SEC   = 1.0         # how often to check for new rows
TIMESTEPS  = 10          # must match training
VOTE_THRES = 2           # need this many models to flag intruder
CONSEC_K   = 3           # require K consecutive anomaly decisions to trigger action

# ---- Actions (stub) ----
def on_intrusion_detected(rows_df: pd.DataFrame):
    # TODO: integrate with your system policy: lock screen, kill session, alert, etc.
    print("🚨 INTRUSION TRIGGERED on rows:", list(rows_df.index.values)[-CONSEC_K:])

# ---- Load models ----
def safe_load(path, loader):
    try:
        return loader(path)
    except Exception:
        return None

scaler = safe_load(os.path.join(MODEL_DIR, "scaler.pkl"), joblib.load)
ocsvm  = safe_load(os.path.join(MODEL_DIR, "oneclass_svm.pkl"), joblib.load)
isof   = safe_load(os.path.join(MODEL_DIR, "isolation_forest.pkl"), joblib.load)
lstm   = None
lstm_thr = None
try:
    lstm = load_model(os.path.join(MODEL_DIR, "lstm_autoencoder.h5"))
    lstm_thr = joblib.load(os.path.join(MODEL_DIR, "lstm_threshold.pkl"))
except Exception:
    pass

# feature order used at train time
with open(os.path.join(MODEL_DIR, "feature_names.json"), "r") as f:
    FEATURE_ORDER = json.load(f)

if scaler is None:
    raise RuntimeError("Missing scaler.pkl. Re-run training.")

# rolling buffer for LSTM sequences (scaled features)
seq_buffer = []
last_row_processed = 0
consec_flags = 0

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # drop non-features if present
    df = df.drop(columns=[c for c in ["Label", "Timestamp"] if c in df.columns], errors="ignore")
    # keep only columns known at training time (missing ones filled with 0)
    for c in FEATURE_ORDER:
        if c not in df.columns:
            df[c] = 0.0
    # drop extras not in FEATURE_ORDER
    df = df[FEATURE_ORDER]
    # fill NaNs with column means per batch (lightweight)
    df = df.fillna(df.mean(numeric_only=True))
    # ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def score_lstm(batch_scaled: np.ndarray) -> np.ndarray:
    """
    Returns per-row anomaly (1/0) aligned with the batch rows.
    Uses seq_buffer + new rows to form sequences.
    """
    global seq_buffer
    preds = np.full(shape=(batch_scaled.shape[0],), fill_value=np.nan)

    # extend buffer
    for row in batch_scaled:
        seq_buffer.append(row)
        if len(seq_buffer) > 2 * TIMESTEPS:  # cap growth
            seq_buffer = seq_buffer[-(2 * TIMESTEPS):]

    # build sequences ending at each new row
    if lstm is None or lstm_thr is None:
        return np.zeros_like(preds)

    for i in range(batch_scaled.shape[0]):
        # sequence ending at current new row = last TIMESTEPS of buffer
        if len(seq_buffer) >= TIMESTEPS:
            seq = np.array(seq_buffer[-TIMESTEPS:])
            seq_in = seq.reshape(1, TIMESTEPS, -1)
            seq_out = lstm.predict(seq_in, verbose=0)
            err = float(np.mean((seq_in - seq_out) ** 2))
            preds[i] = 1.0 if err > float(lstm_thr) else 0.0
        else:
            preds[i] = 0.0  # not enough history yet -> treat as owner
    return preds

def ensemble_vote(df_rows: pd.DataFrame) -> pd.DataFrame:
    X = prepare_features(df_rows)
    X_scaled = scaler.transform(X.values)

    out = pd.DataFrame(index=df_rows.index)
    # One-class voters (1 = intruder)
    if ocsvm is not None:
        out["OneClassSVM"] = (ocsvm.predict(X_scaled) == -1).astype(int)
    else:
        out["OneClassSVM"] = 0

    if isof is not None:
        out["IsolationForest"] = (isof.predict(X_scaled) == -1).astype(int)
    else:
        out["IsolationForest"] = 0

    lstm_votes = score_lstm(X_scaled)
    out["LSTM_AE"] = pd.to_numeric(lstm_votes, errors="coerce").fillna(0).astype(int)

    # vote fuse
    out["VotesAvailable"] = 3  # we always compute 3 here; if some models missing, adjust accordingly
    out["IntruderVotes"] = out[["OneClassSVM", "IsolationForest", "LSTM_AE"]].sum(axis=1)
    out["EnsemblePrediction"] = (out["IntruderVotes"] >= VOTE_THRES).astype(int)
    return out

def tail_loop():
    global last_row_processed, consec_flags
    print("🔎 Monitoring:", DATA_CSV)
    while True:
        try:
            if not os.path.exists(DATA_CSV) or os.path.getsize(DATA_CSV) == 0:
                time.sleep(POLL_SEC)
                continue

            df = pd.read_csv(DATA_CSV)
            if df.empty:
                time.sleep(POLL_SEC)
                continue

            # take only new rows
            new = df.iloc[last_row_processed:]
            if new.empty:
                time.sleep(POLL_SEC)
                continue

            # keep Timestamp for logs (optional)
            timestamps = new["Timestamp"] if "Timestamp" in new.columns else None

            votes = ensemble_vote(new)
            preds = votes["EnsemblePrediction"].values

            # intrusion policy: K consecutive anomalies
            for i, pred in zip(new.index, preds):
                if pred == 1:
                    consec_flags += 1
                else:
                    consec_flags = 0

                if consec_flags >= CONSEC_K:
                    window = df.loc[i - CONSEC_K + 1 : i]
                    if timestamps is not None:
                        print(f"🚩 {CONSEC_K} consecutive anomalies ending at {timestamps.loc[i]}")
                    on_intrusion_detected(window)
                    consec_flags = 0  # reset after action

            # progress
            last_row_processed = len(df)

        except Exception as e:
            print("❌ Monitor error:", e)
        time.sleep(POLL_SEC)

if __name__ == "__main__":
    tail_loop()