#!/usr/bin/env python3
# """
# monitor_online_adaptive.py

# Continuously monitors cd_vector.csv, scores new rows with an ensemble
# (OneClassSVM, IsolationForest, LSTM-AE), triggers intrusion action on
# K consecutive anomalies, and adaptively retrains on accepted owner data.

# Place this file at: Implementation/src/monitor_online_adaptive.py
# """

import os
import time
import json
import joblib
import traceback
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

# tensorflow imports (used only if available)
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

load_dotenv()

# -------------------------
# CONFIG (tweak to taste)
# -------------------------
BASE_DIR = os.getenv("base_dir", "/Users/mannpatel/Desktop/RMS/Implementation")
DATA_CSV = os.path.join(BASE_DIR, "data", "cd_vector.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

POLL_SEC = 1.0          # how often to check file for new rows
TIMESTEPS = 10          # must match training LSTM timesteps
VOTE_THRES = 2          # how many model votes required to mark an intruder (>=)
CONSEC_K = 3            # consecutive intrusions to trigger action
FORGIVE_ROWS = 20       # after intrusion, accept next N rows if not extreme
WINDOW_SIZE = 1500      # sliding owner buffer for retraining
RETRAIN_AFTER = 300     # retrain after this many newly accepted owner rows
MAX_POISON_RATE = 0.3   # if >30% of recent owner-buffer rows look anomalous, skip retrain

# persistence
OFFSET_PATH = os.path.join(MODELS_DIR, "monitor_offset.json")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")  # joblib dump
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
OCSVM_PATH = os.path.join(MODELS_DIR, "oneclass_svm.pkl")
ISO_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_autoencoder.h5")
LSTM_THR_PATH = os.path.join(MODELS_DIR, "lstm_threshold.pkl")

# -------------------------
# Helpers
# -------------------------
def safe_load(path, loader=joblib.load):
    try:
        return loader(path)
    except Exception:
        return None

def save_offset(val):
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(OFFSET_PATH, "w") as f:
            json.dump({"last_row": val}, f)
    except Exception:
        pass

def load_offset():
    try:
        if os.path.exists(OFFSET_PATH):
            with open(OFFSET_PATH, "r") as f:
                return json.load(f).get("last_row", 0)
    except Exception:
        pass
    return 0

def build_lstm_autoencoder(timesteps, n_features):
    model = Sequential([
        LSTM(64, activation="relu", input_shape=(timesteps, n_features), return_sequences=False),
        RepeatVector(timesteps),
        LSTM(64, activation="relu", return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# -------------------------
# Prepare / align features
# -------------------------
def load_feature_order():
    # prefer joblib pickle list; fallback to json or None
    fn = None
    try:
        if os.path.exists(FEATURE_NAMES_PATH):
            fn = joblib.load(FEATURE_NAMES_PATH)
    except Exception:
        try:
            jpath = FEATURE_NAMES_PATH.replace(".pkl", ".json")
            if os.path.exists(jpath):
                with open(jpath, "r") as f:
                    fn = json.load(f)
        except Exception:
            fn = None
    return fn

def prepare_features(df: pd.DataFrame, feature_order):
    # Drop meta
    df = df.copy()
    df = df.drop(columns=[c for c in ["Label", "Timestamp"] if c in df.columns], errors="ignore")

    # If feature_order known, ensure those columns exist (fill missing with 0)
    if feature_order:
        for c in feature_order:
            if c not in df.columns:
                df[c] = 0.0
        df = df[feature_order]  # keep training order
    else:
        # fallback: keep numeric columns only
        df = df.select_dtypes(include=[np.number])

    # Fill numeric NaNs with column mean
    if not df.empty:
        df = df.fillna(df.mean(numeric_only=True))
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return df

# -------------------------
# LSTM scoring helpers
# -------------------------
seq_buffer = deque(maxlen=2 * TIMESTEPS)  # rolling scaled feature rows for LSTM

def align_seq_scores_to_rows(seq_scores, n_rows, timesteps):
    """Map each sequence score (length n_rows - timesteps + 1) to the index where sequence ends.
       Returns array length n_rows with np.nan first (timesteps-1) entries, then seq_scores."""
    out = np.full(shape=(n_rows,), fill_value=np.nan)
    if seq_scores is None or len(seq_scores) == 0:
        return out
    start_idx = timesteps - 1
    out[start_idx:start_idx + len(seq_scores)] = seq_scores
    return out

def score_lstm_rows(lstm_model, lstm_thr, batch_scaled):
    """
    Use seq_buffer + batch_scaled to compute per-row binary (0 owner / 1 intruder).
    """
    n = batch_scaled.shape[0]
    preds = np.zeros(shape=(n,), dtype=int)

    # extend the buffer with batch rows
    for row in batch_scaled:
        seq_buffer.append(row)

    if lstm_model is None or lstm_thr is None:
        return preds, np.full(n, np.nan)

    seqs = []
    # build all sequences that end at each new row in this batch
    for i in range(n):
        if len(seq_buffer) >= TIMESTEPS:
            # last TIMESTEPS in buffer correspond to sequence ending at current new row
            seq = np.array(list(seq_buffer)[-TIMESTEPS:])
            seqs.append(seq)
        else:
            seqs.append(None)

    seqs_valid = np.array([s for s in seqs if s is not None])
    if len(seqs_valid) == 0:
        return preds, np.full(n, np.nan)

    # predict in batches (reshape)
    seqs_in = seqs_valid.reshape((len(seqs_valid), TIMESTEPS, seqs_valid.shape[2]))
    seqs_out = lstm_model.predict(seqs_in, verbose=0)
    seq_err = np.mean((seqs_in - seqs_out) ** 2, axis=(1, 2))

    # map errors back into the per-row array
    err_idx = 0
    row_errs = np.full(n, np.nan)
    for i, s in enumerate(seqs):
        if s is not None:
            e = seq_err[err_idx]
            row_errs[i] = e
            preds[i] = 1 if e > float(lstm_thr) else 0
            err_idx += 1
        else:
            row_errs[i] = np.nan
            preds[i] = 0  # treat early rows as owner

    return preds, row_errs

# -------------------------
# Ensemble vote
# -------------------------
def ensemble_vote(df_batch, scaler, ocsvm, isof, lstm_model, lstm_thr, feature_order):
    # prepare features (aligned + numeric)
    X = prepare_features(df_batch, feature_order)
    if X.empty:
        # return zeros and nans
        out = pd.DataFrame(index=df_batch.index)
        out["OneClassSVM"] = 0
        out["IsolationForest"] = 0
        out["LSTM_AE"] = 0
        out["LSTM_Error"] = np.nan
        out["VotesAvailable"] = 0
        out["IntruderVotes"] = 0
        out["EnsemblePrediction"] = 0
        return out

    # scale
    X_scaled = scaler.transform(X.values)

    out = pd.DataFrame(index=df_batch.index)

    # ocsvm, iso predictions (1 = intruder)
    if ocsvm is not None:
        oc_pred = ocsvm.predict(X_scaled)  # +1 inlier, -1 outlier
        out["OneClassSVM"] = (oc_pred == -1).astype(int)
    else:
        out["OneClassSVM"] = 0

    if isof is not None:
        iso_pred = isof.predict(X_scaled)
        out["IsolationForest"] = (iso_pred == -1).astype(int)
    else:
        out["IsolationForest"] = 0

    # LSTM
    lstm_preds, lstm_errs = score_lstm_rows(lstm_model, lstm_thr, X_scaled)
    out["LSTM_AE"] = lstm_preds
    out["LSTM_Error"] = lstm_errs

    # votes
    out["VotesAvailable"] = (~out[["OneClassSVM", "IsolationForest", "LSTM_AE"]].isna()).sum(axis=1)
    out["IntruderVotes"] = out[["OneClassSVM", "IsolationForest", "LSTM_AE"]].sum(axis=1)
    out["EnsemblePrediction"] = (out["IntruderVotes"] >= VOTE_THRES).astype(int)

    return out

# -------------------------
# Retrain models from owner buffer (safe)
# -------------------------
def retrain_models_from_df(owner_df, feature_order_target=None):
    """
    Train scaler, OCSVM, IsolationForest, LSTM-AE from owner_df.
    owner_df: DataFrame containing feature columns (may include extra columns).
    Returns: dict(paths) or raises exception.
    """
    print("🔁 Retraining models on accepted owner buffer (size=%d) ..." % len(owner_df))
    df = owner_df.copy()
    # drop meta
    df = df.drop(columns=[c for c in ["Label", "Timestamp"] if c in df.columns], errors="ignore")
    # prefer feature_order_target to select columns
    if feature_order_target:
        for c in feature_order_target:
            if c not in df.columns:
                df[c] = 0.0
        df = df[feature_order_target]
    else:
        # infer numeric columns
        df = df.select_dtypes(include=[np.number])

    # clean
    df = df.fillna(df.mean(numeric_only=True))
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        raise RuntimeError("No numeric data to retrain on after cleaning.")

    # fit scaler
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    # One-class models
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(X)
    isof = IsolationForest(contamination=0.05, random_state=42)
    isof.fit(X)

    # LSTM training (only if enough rows)
    if len(X) > TIMESTEPS:
        X_seq = np.array([X[i:i + TIMESTEPS] for i in range(len(X) - TIMESTEPS)])
        n_features = X.shape[1]
        ae = build_lstm_autoencoder(TIMESTEPS, n_features)
        ae.fit(X_seq, X_seq, epochs=15, batch_size=64, validation_split=0.1, verbose=0)
        X_pred = ae.predict(X_seq, verbose=0)
        train_err = np.mean((X_seq - X_pred) ** 2, axis=(1, 2))
        thr = float(np.percentile(train_err, 95))
        # save LSTM
        os.makedirs(MODELS_DIR, exist_ok=True)
        ae.save(LSTM_MODEL_PATH)
        joblib.dump(thr, LSTM_THR_PATH)
        print("  - LSTM retrained, thr=", thr)
    else:
        # remove old LSTM if exists? keep existing
        ae = None
        thr = None
        print("  - Not enough rows for LSTM retrain (need > TIMESTEPS)")

    # persist scaler & models
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(ocsvm, OCSVM_PATH)
    joblib.dump(isof, ISO_PATH)

    # persist feature order
    feature_names = df.columns.tolist()
    joblib.dump(feature_names, FEATURE_NAMES_PATH)

    print("✅ Retrain complete and saved to models/")

    return {
        "scaler": SCALER_PATH,
        "ocsvm": OCSVM_PATH,
        "isof": ISO_PATH,
        "lstm": LSTM_MODEL_PATH if ae is not None else None,
        "lstm_thr": LSTM_THR_PATH if thr is not None else None,
        "feature_names": FEATURE_NAMES_PATH
    }

# -------------------------
# Main monitor loop
# -------------------------
def on_intrusion_detected(window_df: pd.DataFrame):
    # Reactive policy: here is where you implement lock-screen, notifications, etc.
    print("🚨 INTRUSION ACTION: rows", list(window_df.index)[-CONSEC_K:])
    # Example stub: write to a log file
    log_path = os.path.join(BASE_DIR, "data", "intrusion_log.txt")
    with open(log_path, "a") as f:
        f.write(f"{datetime.now().isoformat()} - Intrusion detected - rows {list(window_df.index)[-CONSEC_K:]}\n")

def monitor_loop():
    # load models and state
    os.makedirs(MODELS_DIR, exist_ok=True)
    # offset for persistence
    last_row = load_offset()
    # owner buffer stores accepted owner rows (DataFrame)
    owner_buffer = pd.DataFrame()
    accepted_since_retrain = 0
    in_intrusion_mode = False
    forgive_remaining = 0
    consec_flags = 0

    # load model artifacts (if present)
    scaler = safe_load(SCALER_PATH, joblib.load)
    ocsvm = safe_load(OCSVM_PATH, joblib.load)
    isof = safe_load(ISO_PATH, joblib.load)
    lstm_model = None
    lstm_thr = None
    try:
        if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_THR_PATH):
            lstm_model = load_model(LSTM_MODEL_PATH)
            lstm_thr = joblib.load(LSTM_THR_PATH)
    except Exception:
        lstm_model, lstm_thr = None, None

    feature_order = load_feature_order()

    print("👀 Monitoring", DATA_CSV)
    print(f"   Window={WINDOW_SIZE}, Retrain every {RETRAIN_AFTER} accepted rows, K={CONSEC_K}, Vote≥{VOTE_THRES}")

    # If models missing but enough data exists, attempt an initial train using all existing cd_vector rows.
    try:
        if (scaler is None or ocsvm is None or isof is None) and os.path.exists(DATA_CSV):
            df_all = pd.read_csv(DATA_CSV)
            if len(df_all) > max(TIMESTEPS, 500):  # heuristic: need some data
                print("⚠️ Models missing; doing initial train on existing data...")
                retrain_models_from_df(df_all.tail(WINDOW_SIZE), feature_order_target=None)
                scaler = joblib.load(SCALER_PATH)
                ocsvm = joblib.load(OCSVM_PATH)
                isof = joblib.load(ISO_PATH)
                if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_THR_PATH):
                    lstm_model = load_model(LSTM_MODEL_PATH)
                    lstm_thr = joblib.load(LSTM_THR_PATH)
                feature_order = load_feature_order()

    except Exception as e:
        print("Error during initial model check:", e, traceback.format_exc())

    # main polling loop
    while True:
        try:
            if not os.path.exists(DATA_CSV) or os.path.getsize(DATA_CSV) == 0:
                time.sleep(POLL_SEC)
                continue

            df = pd.read_csv(DATA_CSV)
            if df.empty:
                time.sleep(POLL_SEC)
                continue

            total_rows = len(df)
            if last_row >= total_rows:
                time.sleep(POLL_SEC)
                continue

            # new rows to process
            new = df.iloc[last_row:total_rows].reset_index(drop=True)
            if new.empty:
                last_row = total_rows
                save_offset(last_row)
                time.sleep(POLL_SEC)
                continue

            # score batch
            if scaler is None:
                print("⚠️ No scaler available — marking rows as owner until scaler exists.")
                votes = pd.DataFrame(index=new.index)
                votes["EnsemblePrediction"] = 0
                votes["IntruderVotes"] = 0
            else:
                votes = ensemble_vote(new, scaler, ocsvm, isof, lstm_model, lstm_thr, feature_order)

            # per-row policy
            for idx_local in range(len(new)):
                row_votes = int(votes["EnsemblePrediction"].iat[idx_local])
                intr_votes = int(votes["IntruderVotes"].iat[idx_local]) if "IntruderVotes" in votes.columns else 0

                # If currently in forgiveness/adaptation after intrusion
                if in_intrusion_mode and forgive_remaining > 0:
                    # accept row if not strongly anomalous (e.g., intruder votes < VOTE_THRES)
                    accept_row = intr_votes < VOTE_THRES
                    forgive_remaining -= 1
                    if accept_row:
                        # append to owner buffer
                        row_df = new.iloc[[idx_local]]
                        owner_buffer = pd.concat([owner_buffer, row_df], ignore_index=True)
                        accepted_since_retrain += 1
                    else:
                        # still intruder; do not accept
                        pass

                else:
                    # normal mode: accept only rows flagged as owner (ensemble=0)
                    if row_votes == 0:
                        row_df = new.iloc[[idx_local]]
                        owner_buffer = pd.concat([owner_buffer, row_df], ignore_index=True)
                        accepted_since_retrain += 1
                        consec_flags = 0
                    else:
                        consec_flags += 1
                        # if consecutive flagged, trigger intrusion action
                        if consec_flags >= CONSEC_K:
                            # intrusion detected: handle it, enable forgiveness mode
                            start_idx = last_row + idx_local - CONSEC_K + 1
                            end_idx = last_row + idx_local
                            window = df.iloc[max(0, start_idx):end_idx + 1]
                            on_intrusion_detected(window)
                            in_intrusion_mode = True
                            forgive_remaining = FORGIVE_ROWS
                            consec_flags = 0
                            # do not accept these anomaly rows into owner buffer
                            # continue to next rows

                # maintain sliding window size
                if len(owner_buffer) > WINDOW_SIZE:
                    owner_buffer = owner_buffer.tail(WINDOW_SIZE).reset_index(drop=True)

            # Retrain if enough accepted rows accumulated
            if accepted_since_retrain >= RETRAIN_AFTER and len(owner_buffer) >= TIMESTEPS + 10:
                # check poisoning: what fraction of recent owner_buffer rows are anomaly by current models?
                try:
                    # prepare last N rows
                    recent = owner_buffer.tail(min(len(owner_buffer), WINDOW_SIZE))
                    prep = prepare_features(recent, feature_order)
                    X_scaled = joblib.load(SCALER_PATH).transform(prep.values) if os.path.exists(SCALER_PATH) else None
                    if X_scaled is not None:
                        # use ocsvm/isof if present
                        bad = 0
                        total_check = X_scaled.shape[0]
                        if ocsvm is not None:
                            bad += (ocsvm.predict(X_scaled) == -1).sum()
                        if isof is not None:
                            bad += (isof.predict(X_scaled) == -1).sum()
                        # rough anomaly rate (double counting possible) -> normalize
                        poison_rate = (bad / (2 * total_check)) if total_check > 0 else 0.0
                    else:
                        poison_rate = 0.0
                except Exception:
                    poison_rate = 1.0

                if poison_rate <= MAX_POISON_RATE:
                    try:
                        retrain_models_from_df(owner_buffer.tail(WINDOW_SIZE), feature_order_target=feature_order)
                        # reload models and scaler
                        scaler = joblib.load(SCALER_PATH)
                        ocsvm = joblib.load(OCSVM_PATH)
                        isof = joblib.load(ISO_PATH)
                        if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_THR_PATH):
                            lstm_model = load_model(LSTM_MODEL_PATH)
                            lstm_thr = joblib.load(LSTM_THR_PATH)
                        # reset counter
                        accepted_since_retrain = 0
                        in_intrusion_mode = False
                        print("🔁 Retrain triggered and completed.")
                    except Exception as e:
                        print("⚠️ Retrain failed:", e)
                else:
                    print(f"⚠️ Skipping retrain due to high poison rate ({poison_rate:.2f})")
                    # reset accepted counter to avoid immediate repeat
                    accepted_since_retrain = 0

            # persist last_row
            last_row = total_rows
            save_offset(last_row)

        except KeyboardInterrupt:
            print("Monitor stopped by user.")
            break
        except Exception as e:
            print("Monitor error:", e)
            traceback.print_exc()

        time.sleep(POLL_SEC)

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    monitor_loop()