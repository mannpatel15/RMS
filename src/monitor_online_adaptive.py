import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =======================
# Config
# =======================
load_dotenv()
BASE_DIR   = os.getenv("base_dir")
DATA_CSV   = os.path.join(BASE_DIR, "data", "cd_vector.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# Streaming + adaptation
POLL_SEC        = 1.0     # loop polling interval
TIMESTEPS       = 10      # same for train/predict
WINDOW_ROWS     = 1500    # max rows kept as "owner window"
MIN_WINDOW_ROWS = 300     # min rows required to (re)train
RETRAIN_EVERY   = 300     # retrain after this many accepted (owner) rows
COOLDOWN_AFTER_ALERT_ROWS = 50  # pause updates after alert

# Decisions
VOTE_THRES      = 2       # 2 of 3 models must flag to count as anomaly
CONSEC_K        = 3       # require K consecutive anomalies to trigger alert
LSTM_Q_PCT      = 97.5    # LSTM threshold percentile on recent window errors
OCSVM_NU        = 0.05    # expected anomaly rate for OCSVM
IF_CONTAM       = 0.05    # expected anomaly rate for IsolationForest

# =======================
# Utilities
# =======================

def build_lstm_autoencoder(timesteps, n_features):
    model = Sequential([
        LSTM(64, activation="relu", input_shape=(timesteps, n_features), return_sequences=False),
        RepeatVector(timesteps),
        LSTM(64, activation="relu", return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return model

def safe_load(path, loader):
    try:
        return loader(path)
    except Exception:
        return None

def save_feature_order(cols):
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(cols, f)

def load_feature_order():
    try:
        with open(os.path.join(MODEL_DIR, "feature_names.json"), "r") as f:
            return json.load(f)
    except Exception:
        return None

def prepare_features(df: pd.DataFrame, feature_order=None) -> pd.DataFrame:
    # drop non-features
    df = df.drop(columns=[c for c in ["Label", "Timestamp"] if c in df.columns], errors="ignore")
    # keep only numeric
    df = df.select_dtypes(include=[np.number])
    # fill NaNs with column means
    df = df.fillna(df.mean(numeric_only=True))

    if feature_order is not None:
        # add missing as 0.0
        for c in feature_order:
            if c not in df.columns:
                df[c] = 0.0
        # drop extra
        df = df[feature_order]

    # ensure finite
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def make_sequences(X_scaled: np.ndarray, timesteps: int) -> np.ndarray:
    if len(X_scaled) <= timesteps:
        return np.empty((0, timesteps, X_scaled.shape[1]))
    return np.array([X_scaled[i:i+timesteps] for i in range(len(X_scaled) - timesteps)])

def lstm_errors(model, X_seq):
    X_pred = model.predict(X_seq, verbose=0)
    return np.mean((X_seq - X_pred) ** 2, axis=(1, 2))

# =======================
# Intrusion action hook
# =======================
def on_intrusion_detected(rows_df: pd.DataFrame):
    # TODO: integrate your policy (lock screen/logout/alert/etc.)
    tail_times = rows_df["Timestamp"].tail(CONSEC_K).tolist() if "Timestamp" in rows_df.columns else ["<no-ts>"]
    print(f"🚨 INTRUSION: {CONSEC_K} consecutive anomalies. Tail timestamps: {tail_times}")

# =======================
# State
# =======================
class OnlineState:
    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)

        self.scaler = safe_load(os.path.join(MODEL_DIR, "scaler.pkl"), joblib.load)
        self.ocsvm  = safe_load(os.path.join(MODEL_DIR, "oneclass_svm.pkl"), joblib.load)
        self.isof   = safe_load(os.path.join(MODEL_DIR, "isolation_forest.pkl"), joblib.load)
        self.ae     = None
        self.ae_thr = None

        try:
            self.ae     = load_model(os.path.join(MODEL_DIR, "lstm_autoencoder.h5"))
            self.ae_thr = joblib.load(os.path.join(MODEL_DIR, "lstm_threshold.pkl"))
        except Exception:
            pass

        self.feature_order = load_feature_order()

        self.window_df = pd.DataFrame()   # owner window (raw)
        self.accepted_since_retrain = 0   # accepted owner rows since last retrain
        self.seq_buffer = []              # for per-row LSTM scoring
        self.last_row_processed = 0
        self.consec_flags = 0
        self.cooldown_left = 0

    def fitted(self):
        return (self.scaler is not None) and (self.ocsvm is not None) and (self.isof is not None) and (self.ae is not None) and (self.ae_thr is not None) and (self.feature_order is not None)

    def update_seq_buffer(self, scaled_row):
        self.seq_buffer.append(scaled_row)
        if len(self.seq_buffer) > 2 * TIMESTEPS:
            self.seq_buffer = self.seq_buffer[-(2 * TIMESTEPS):]

# =======================
# Training / Retraining
# =======================
def retrain_on_window(st: OnlineState):
    if len(st.window_df) < MIN_WINDOW_ROWS:
        return False

    # Prepare window features + define feature order if new
    if st.feature_order is None:
        feat_df = prepare_features(st.window_df, None)
        st.feature_order = feat_df.columns.tolist()
        save_feature_order(st.feature_order)
    else:
        feat_df = prepare_features(st.window_df, st.feature_order)

    X = feat_df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit one-class models
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=OCSVM_NU)
    ocsvm.fit(X_scaled)

    isof = IsolationForest(contamination=IF_CONTAM, random_state=42)
    isof.fit(X_scaled)

    # LSTM-AE train
    X_seq = make_sequences(X_scaled, TIMESTEPS)
    if len(X_seq) == 0:
        return False

    ae = build_lstm_autoencoder(TIMESTEPS, X_scaled.shape[1])
    ae.fit(X_seq, X_seq, epochs=8, batch_size=64, validation_split=0.1, shuffle=True, verbose=0)

    errs = lstm_errors(ae, X_seq)
    thr = float(np.percentile(errs, LSTM_Q_PCT))

    # Persist
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(ocsvm,  os.path.join(MODEL_DIR, "oneclass_svm.pkl"))
    joblib.dump(isof,   os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    ae.save(os.path.join(MODEL_DIR, "lstm_autoencoder.h5"))
    joblib.dump(thr, os.path.join(MODEL_DIR, "lstm_threshold.pkl"))
    save_feature_order(st.feature_order)

    # Swap into state
    st.scaler, st.ocsvm, st.isof, st.ae, st.ae_thr = scaler, ocsvm, isof, ae, thr
    st.accepted_since_retrain = 0
    print(f"✅ (Re)trained on window: n={len(st.window_df)}, thr={thr:.6f}")
    return True

# =======================
# Scoring
# =======================
def score_batch(st: OnlineState, raw_rows: pd.DataFrame) -> pd.DataFrame:
    # Prepare + scale
    X_df = prepare_features(raw_rows, st.feature_order)
    if st.scaler is None or st.feature_order is None or X_df.empty:
        # Not fitted yet, default to "owner" (0) until we can train
        res = pd.DataFrame(index=raw_rows.index)
        res["OneClassSVM"] = 0
        res["IsolationForest"] = 0
        res["LSTM_AE"] = 0
        res["VotesAvailable"] = 0
        res["IntruderVotes"] = 0
        res["EnsemblePrediction"] = 0
        return res

    X_scaled = st.scaler.transform(X_df.values)

    out = pd.DataFrame(index=raw_rows.index)

    # OCSVM / IF
    out["OneClassSVM"] = (st.ocsvm.predict(X_scaled) == -1).astype(int) if st.ocsvm is not None else 0
    out["IsolationForest"] = (st.isof.predict(X_scaled) == -1).astype(int) if st.isof is not None else 0

    # LSTM per-row using rolling buffer
    lstm_votes = []
    for r in X_scaled:
        st.update_seq_buffer(r)
        if st.ae is None or st.ae_thr is None or len(st.seq_buffer) < TIMESTEPS:
            lstm_votes.append(0)
            continue
        seq = np.array(st.seq_buffer[-TIMESTEPS:])
        seq = seq.reshape(1, TIMESTEPS, -1)
        pred = st.ae.predict(seq, verbose=0)
        err = float(np.mean((seq - pred) ** 2))
        lstm_votes.append(1 if err > float(st.ae_thr) else 0)
    out["LSTM_AE"] = np.array(lstm_votes, dtype=int)

    out["VotesAvailable"] = 3
    out["IntruderVotes"] = out[["OneClassSVM", "IsolationForest", "LSTM_AE"]].sum(axis=1)
    out["EnsemblePrediction"] = (out["IntruderVotes"] >= VOTE_THRES).astype(int)
    return out

# =======================
# Main loop
# =======================
def main():
    st = OnlineState()
    print(f"👀 Monitoring {DATA_CSV}")
    print(f"   Window={WINDOW_ROWS}, Retrain every {RETRAIN_EVERY} accepted rows, K={CONSEC_K}, Vote≥{VOTE_THRES}")

    while True:
        try:
            if not os.path.exists(DATA_CSV) or os.path.getsize(DATA_CSV) == 0:
                time.sleep(POLL_SEC)
                continue

            df_all = pd.read_csv(DATA_CSV)
            if df_all.empty:
                time.sleep(POLL_SEC)
                continue

            # New rows since last pass
            new = df_all.iloc[st.last_row_processed:]
            if new.empty:
                time.sleep(POLL_SEC)
                continue

            # If we have no window yet, bootstrap it with earliest rows
            if st.window_df.empty:
                # Start with earliest chunk; DO NOT assume they’re all owner-like.
                # We’ll rapidly retrain as better “accepted” rows accumulate.
                st.window_df = new.copy().tail(MIN_WINDOW_ROWS).reset_index(drop=True)

            # Train if not fitted and we have enough window
            if not st.fitted() and len(st.window_df) >= MIN_WINDOW_ROWS:
                retrain_on_window(st)

            # Score new rows
            votes = score_batch(st, new)

            # Trigger logic
            for i, pred in zip(new.index, votes["EnsemblePrediction"].values):
                if pred == 1:
                    st.consec_flags += 1
                else:
                    st.consec_flags = 0

                if st.consec_flags >= CONSEC_K:
                    window_slice = df_all.loc[i - CONSEC_K + 1: i]
                    on_intrusion_detected(window_slice)
                    st.consec_flags = 0
                    st.cooldown_left = COOLDOWN_AFTER_ALERT_ROWS  # pause window updates

            # Owner-window maintenance:
            # Only add rows predicted as OWNER (0) when not in cooldown.
            if st.cooldown_left > 0:
                st.cooldown_left -= len(new)
            else:
                owner_mask = votes["EnsemblePrediction"].values == 0
                accepted = new.loc[owner_mask]
                if not accepted.empty:
                    st.window_df = pd.concat([st.window_df, accepted], ignore_index=True)
                    # cap window size
                    if len(st.window_df) > WINDOW_ROWS:
                        st.window_df = st.window_df.iloc[-WINDOW_ROWS:].reset_index(drop=True)
                    st.accepted_since_retrain += len(accepted)

            # Periodic retrain on evolving owner window
            if st.accepted_since_retrain >= RETRAIN_EVERY and len(st.window_df) >= MIN_WINDOW_ROWS:
                retrain_on_window(st)

            # Move forward
            st.last_row_processed = len(df_all)

        except Exception as e:
            print("❌ Monitor error:", e)

        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()