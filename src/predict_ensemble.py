# Implementation/src/predict_ensemble.py

import os
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

load_dotenv()
base_dir = os.getenv("base_dir")
MODEL_DIR = os.path.join(base_dir, "models")
DATA_DIR = os.path.join(base_dir, "data")

# --- config ---
TIMESTEPS = 10  # must match what you used during LSTM training
INPUT_CSV = os.path.join(DATA_DIR, "cd_vector.csv")  # you can point this to any CSV of vectors

def _safe_load(path, loader):
    try:
        return loader(path)
    except Exception:
        return None

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # Keep numeric features only; drop non-numeric and non-feature columns
    drop_cols = [c for c in ["Label", "Timestamp"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")
    # Fill numeric NaNs with column mean
    df = df.fillna(df.mean(numeric_only=True))
    # Keep only numeric dtypes
    df = df.select_dtypes(include=[np.number])
    return df

def _sequences_from_matrix(X_scaled: np.ndarray, timesteps: int) -> np.ndarray:
    if len(X_scaled) < timesteps:
        return np.empty((0, timesteps, X_scaled.shape[1]))
    return np.array([X_scaled[i:i+timesteps] for i in range(len(X_scaled) - timesteps)])

def _align_seq_scores_to_rows(seq_scores: np.ndarray, n_rows: int, timesteps: int) -> np.ndarray:
    """
    seq_scores has length n_rows - timesteps.
    Map each sequence score to the row where the sequence ends.
    Fill the first (timesteps) rows with NaN so array length == n_rows.
    """
    out = np.full(shape=(n_rows,), fill_value=np.nan, dtype=float)
    if len(seq_scores) > 0:
        out[timesteps:] = seq_scores
    return out

def predict_ensemble(input_csv: str = INPUT_CSV, vote_threshold: int = 2) -> pd.DataFrame:
    # 1) Load data
    raw = pd.read_csv(input_csv)
    X_df = _prepare_features(raw)
    n_rows = len(X_df)

    if n_rows == 0:
        raise ValueError("No rows to score after preprocessing.")

    # 2) Load scaler
    scaler = _safe_load(os.path.join(MODEL_DIR, "scaler.pkl"), joblib.load)
    if scaler is None:
        raise FileNotFoundError("Missing scaler.pkl in models/. Please train models first.")

    X_scaled = scaler.transform(X_df.values)

    # 3) Prepare outputs container
    outputs = pd.DataFrame(index=raw.index)
    # Keep original Timestamp/Label if present for reference
    if "Timestamp" in raw.columns:
        outputs["Timestamp"] = raw["Timestamp"]
    if "Label" in raw.columns:
        outputs["Label"] = raw["Label"]

    # 4) One-Class SVM
    ocsvm = _safe_load(os.path.join(MODEL_DIR, "oneclass_svm.pkl"), joblib.load)
    if ocsvm is not None:
        oc_pred = ocsvm.predict(X_scaled)  # +1 = inlier (owner), -1 = outlier (intruder)
        outputs["OneClassSVM"] = (oc_pred == -1).astype(int)  # 1 = intruder
    else:
        outputs["OneClassSVM"] = np.nan

    # 5) Isolation Forest
    iso = _safe_load(os.path.join(MODEL_DIR, "isolation_forest.pkl"), joblib.load)
    if iso is not None:
        iso_pred = iso.predict(X_scaled)  # +1 inlier, -1 outlier
        outputs["IsolationForest"] = (iso_pred == -1).astype(int)
    else:
        outputs["IsolationForest"] = np.nan

    # 6) LSTM Autoencoder
    lstm = None
    lstm_thr = None
    try:
        lstm = load_model(os.path.join(MODEL_DIR, "lstm_autoencoder.h5"))
        lstm_thr = joblib.load(os.path.join(MODEL_DIR, "lstm_threshold.pkl"))
    except Exception:
        pass

    if lstm is not None and lstm_thr is not None:
        X_seq = _sequences_from_matrix(X_scaled, TIMESTEPS)
        if len(X_seq) == 0:
            lstm_preds = np.full(shape=(n_rows,), fill_value=np.nan)
        else:
            X_pred = lstm.predict(X_seq, verbose=0)
            seq_err = np.mean(np.power(X_seq - X_pred, 2), axis=(1, 2))
            row_err = _align_seq_scores_to_rows(seq_err, n_rows, TIMESTEPS)
            lstm_preds = (row_err > float(lstm_thr)).astype(float)  # 1=intruder
        outputs["LSTM_AE_Error"] = row_err if lstm is not None and len(X_seq) > 0 else np.nan
        outputs["LSTM_AE"] = lstm_preds
    else:
        outputs["LSTM_AE_Error"] = np.nan
        outputs["LSTM_AE"] = np.nan

    # 7) Optional supervised models (only used if present)
    supervised = {
        "RandomForest": "random_forest.pkl",
        "LogisticRegression": "logistic_regression.pkl",
        "KNN": "knn.pkl",
        "DecisionTree": "decision_tree.pkl",
        # "SVM": "svm.pkl",  # keep off for now as requested
    }
    for name, fname in supervised.items():
        path = os.path.join(MODEL_DIR, fname)
        model = _safe_load(path, joblib.load)
        if model is not None:
            try:
                # Expect 0=owner, 1=intruder for supervised
                outputs[name] = model.predict(X_scaled)
            except Exception:
                outputs[name] = np.nan
        else:
            outputs[name] = np.nan

    # 8) Ensemble vote across available model columns
    voter_cols = [c for c in outputs.columns if c not in ["Timestamp", "Label", "LSTM_AE_Error"]]
    # Convert to numeric 0/1 with NaNs ignored in vote
    vote_matrix = outputs[voter_cols].apply(pd.to_numeric, errors="coerce")

    # Count how many models vote "intruder" (==1) per row
    intruder_votes = (vote_matrix == 1).sum(axis=1, skipna=True)
    # Count how many model votes are present per row
    available_votes = vote_matrix.notna().sum(axis=1)

    # Majority decision: need at least 2 votes by default (tune via vote_threshold)
    # If <2 models available for a row (e.g., early rows due to LSTM timesteps), we still do best-effort majority.
    outputs["VotesAvailable"] = available_votes
    outputs["IntruderVotes"] = intruder_votes
    thresholds = np.minimum(vote_threshold, available_votes)
    outputs["EnsemblePrediction"] = (intruder_votes >= thresholds).astype(int)

    # 9) Quick summary
    total = len(outputs)
    intruders = int(outputs["EnsemblePrediction"].sum())
    print(f"✅ Scored {total} rows | Ensemble flagged {intruders} as intruder(s).")

    # 10) Save results next to input
    out_csv = os.path.join(DATA_DIR, "ensemble_predictions.csv")
    outputs.to_csv(out_csv, index=False)
    print(f"📄 Saved detailed predictions to: {out_csv}")

    return outputs

if __name__ == "__main__":
    results = predict_ensemble(INPUT_CSV)
    # Show a few recent rows (most aligned with LSTM)
    print(results.tail(10))