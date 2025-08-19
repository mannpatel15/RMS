# Implementation/src/predict_ensemble.py
import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv("base_dir")
DATA_DIR = os.path.join(base_dir, "data")
MODEL_DIR = os.path.join(base_dir, "models")

INPUT_CSV = os.path.join(DATA_DIR, "cd_vector_clean.csv")
TIMESTEPS = 10
DEFAULT_VOTE_THRESHOLD = 2  # need >=2 model votes to flag

def _prepare_matrix(raw: pd.DataFrame, feature_names):
    X = raw.copy()
    for col in ["Label", "Timestamp"]:
        if col in X.columns:
            X = X.drop(columns=[col])
    X = X.reindex(columns=feature_names)
    X = X.fillna(X.mean(numeric_only=True))
    return X

def _seqs(X, t):
    if len(X) <= t:
        return np.empty((0, t, X.shape[1]))
    return np.array([X[i:i+t] for i in range(len(X) - t)])

def _align_seq_scores(scores, n_rows, t):
    out = np.full(shape=(n_rows,), fill_value=np.nan, dtype=float)
    if len(scores) > 0:
        out[t:] = scores
    return out

def predict_ensemble(input_csv=INPUT_CSV, vote_threshold=DEFAULT_VOTE_THRESHOLD):
    raw = pd.read_csv(input_csv)

    # Load preprocessing assets
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    X_df = _prepare_matrix(raw, feature_names)
    X = scaler.transform(X_df.values)
    n = len(X)

    out = pd.DataFrame(index=raw.index)
    if "Timestamp" in raw.columns: out["Timestamp"] = raw["Timestamp"]
    if "Label" in raw.columns:     out["Label"] = raw["Label"]

    # One-Class SVM
    try:
        ocsvm = joblib.load(os.path.join(MODEL_DIR, "oneclass_svm.pkl"))
        oc = (ocsvm.predict(X) == -1).astype(int)
        out["OneClassSVM"] = oc
    except Exception:
        out["OneClassSVM"] = np.nan

    # Isolation Forest
    try:
        iso = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
        ifo = (iso.predict(X) == -1).astype(int)
        out["IsolationForest"] = ifo
    except Exception:
        out["IsolationForest"] = np.nan

    # LSTM-AE
    try:
        ae = load_model(os.path.join(MODEL_DIR, "lstm_autoencoder.h5"))
        thr = float(joblib.load(os.path.join(MODEL_DIR, "lstm_threshold.pkl")))
        X_seq = _seqs(X, TIMESTEPS)
        if len(X_seq) == 0:
            err = np.full(shape=(n,), fill_value=np.nan)
            lstm_vote = np.full(shape=(n,), fill_value=np.nan)
        else:
            X_pred = ae.predict(X_seq, verbose=0)
            seq_err = np.mean((X_seq - X_pred) ** 2, axis=(1, 2))
            err = _align_seq_scores(seq_err, n, TIMESTEPS)
            lstm_vote = (err > thr).astype(float)
        out["LSTM_AE_Error"] = err
        out["LSTM_AE"] = lstm_vote
    except Exception:
        out["LSTM_AE_Error"] = np.nan
        out["LSTM_AE"] = np.nan

    # Voting
    voter_cols = [c for c in out.columns if c not in ["Timestamp", "Label", "LSTM_AE_Error"]]
    votes = out[voter_cols].apply(pd.to_numeric, errors="coerce")
    intruder_votes = (votes == 1).sum(axis=1, skipna=True)
    available = votes.notna().sum(axis=1)

    out["VotesAvailable"] = available
    out["IntruderVotes"] = intruder_votes
    thresh = np.minimum(vote_threshold, available)  # per-row threshold
    out["EnsemblePrediction"] = (intruder_votes >= thresh).astype(int)

    save_path = os.path.join(DATA_DIR, "ensemble_predictions.csv")
    out.to_csv(save_path, index=False)
    print(f"✅ Scored {len(out)} rows | Flagged {int(out['EnsemblePrediction'].sum())} intruder(s).")
    print(f"📄 Saved: {save_path}")
    return out

if __name__ == "__main__":
    res = predict_ensemble(INPUT_CSV)
    print(res.tail(10))