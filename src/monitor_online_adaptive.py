# # monitor_online_adaptive.py
# import os
# import json
# import time
# import joblib
# import pandas as pd
# from pathlib import Path
# from sklearn.ensemble import IsolationForest
# from dotenv import load_dotenv

# # ======================
# # Load ENV
# # ======================
# load_dotenv()
# BASE_DIR = os.getenv("BASE_DIR", str(Path(__file__).resolve().parent.parent))
# MODEL_DIR = os.path.join(BASE_DIR, "models")
# DATA_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
# STATE_FILE = os.path.join(BASE_DIR, "cua_state.json")
# print(MODEL_DIR)
# # ======================
# # Load Model
# # ======================
# import os
# import sys

# # Absolute path to models directory

# # Required models
# required_models = [
#     "isolation_forest.pkl",
#     "lof.pkl",
#     "oneclass_svm.pkl",
#     "elliptic_envelope.pkl",
#     "rf_model.pkl",
#     "scaler.pkl",
#     "imputer.pkl",
#     "feature_names.pkl",   # change from json to pkl
#     "training_meta.json"
# ]

# # Check if all required models exist
# missing = [m for m in required_models if not os.path.exists(os.path.join(MODEL_DIR, m))]
# print("DEBUG: Checking models in", MODEL_DIR)
# print("DEBUG: Files available:", os.listdir(MODEL_DIR))
# if missing:
#     print(f"[WARN] Missing required models: {missing}")
#     print("Please run train_models.py first to generate them.")
#     sys.exit(1)
# else:
#     print(f"[INFO] All required models found in {MODEL_DIR}")
    

    
# def load_model():
#     import os, pickle, json
#     from tensorflow.keras.models import load_model as keras_load

#     model_dir = "/Users/mannpatel/Desktop/RMS/Implementation/models"

#     models = {}

#     # Example: Random Forest
#     rf_path = os.path.join(model_dir, "rf_model.pkl")
#     if os.path.exists(rf_path):
#         with open(rf_path, "rb") as f:
#             models["rf"] = pickle.load(f)

#     # Example: Isolation Forest
#     if_path = os.path.join(model_dir, "isolation_forest.pkl")
#     if os.path.exists(if_path):
#         with open(if_path, "rb") as f:
#             models["isolation_forest"] = pickle.load(f)

#     # Example: LSTM Autoencoder
#     lstm_path = os.path.join(model_dir, "lstm_autoencoder.keras")
#     if os.path.exists(lstm_path):
#         models["lstm_autoencoder"] = keras_load(lstm_path)

#     # Meta info (optional)
#     meta_path = os.path.join(model_dir, "training_meta.json")
#     if os.path.exists(meta_path):
#         with open(meta_path, "r") as f:
#             models["meta"] = json.load(f)

#     if not models:
#         return None  # truly nothing found
#     return models

# # ======================
# # Load State
# # ======================
# def load_state():
#     if os.path.exists(STATE_FILE):
#         with open(STATE_FILE, "r") as f:
#             return json.load(f)
#     return {"trained_rows": 0, "retrain_threshold": 200}

# def save_state(state):
#     with open(STATE_FILE, "w") as f:
#         json.dump(state, f, indent=4)

# # ======================
# # Capture user activity (dummy example, replace with real keylog/app usage)
# # ======================
# def capture_user_activity():
#     # Example: collect session features
#     return {
#         "keystroke_speed": 2.1,
#         "mouse_movement": 0.9,
#         "active_app": 3,  # encoded app id
#     }

# # ======================
# # Adaptive Training
# # ======================
# def retrain_model(data):
#     print("[INFO] Retraining model with new data...")
#     model = IsolationForest(
#         n_estimators=150, max_samples="auto", contamination=0.05, random_state=42
#     )
#     model.fit(data)
#     model_path = os.path.join(MODEL_DIR, "cua_model.pkl")
#     joblib.dump(model, model_path)
#     print("[INFO] Model retrained and saved.")
#     return model

# # ======================
# # Main Monitor Loop
# # ======================
# def main():
#     model = load_model()
#     if not model:
#         print("[WARN] No pre-trained model found. Please run train_models.py first.")
#         return

#     state = load_state()
#     print(f"[INFO] Starting CUA monitor. Rows already trained: {state['trained_rows']}")

#     buffer = []  # holds new rows until retrain

#     while True:
#         activity = capture_user_activity()
#         df = pd.DataFrame([activity])

#         # Predict
#         prediction = model.predict(df)[0]  # 1 = owner, -1 = intruder
#         if prediction == 1:
#             print("[OK] Owner detected")
#             buffer.append(activity)
#         else:
#             print("[ALERT] Intruder detected!")

#         # Retrain when buffer large enough
#         if len(buffer) >= state["retrain_threshold"]:
#             df_new = pd.DataFrame(buffer)
#             model = retrain_model(df_new)
#             state["trained_rows"] += len(buffer)
#             save_state(state)
#             buffer = []

#         time.sleep(2)  # poll every 2 sec

# if __name__ == "__main__":
#     main()



# import os
# import json
# import time
# import pickle
# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow.keras as keras
# from collections import deque

# # --- Constants and File Paths ---
# STATE_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cua_state.json"))
# CD_VECTOR_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cd_vector.csv"))
# MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# # ------------------------------------------------------------------
# # Model and State Loading Functions
# # ------------------------------------------------------------------

# def load_models(model_dir):
#     """Loads all model artifacts from the specified directory."""
#     models = {}
#     print(f"DEBUG: Loading models from {model_dir}")
#     for fname in os.listdir(model_dir):
#         fpath = os.path.join(model_dir, fname)
#         key = fname.split(".")[0]

#         try:
#             if fname.endswith(".pkl"):
#                 models[key] = joblib.load(fpath)
#             elif fname.endswith(".json"):
#                 with open(fpath, "r") as f:
#                     models[key] = json.load(f)
#             elif fname.endswith(".npy"):
#                 models[key] = np.load(fpath, allow_pickle=True)
#             elif fname.endswith(".keras"):
#                 models[key] = keras.models.load_model(fpath)
#         except Exception as e:
#             print(f"[WARN] Could not load {fname}: {e}")

#     print("[INFO] Models loaded:", list(models.keys()))
#     return models

# def load_state():
#     """Loads the state from the JSON file. Exits if not found."""
#     if os.path.exists(STATE_FILE):
#         with open(STATE_FILE, "r") as f:
#             return json.load(f)
#     else:
#         # If the state file doesn't exist, we cannot proceed.
#         print("[ERROR] The state file 'cua_state.json' was not found.")
#         print("Please run train_models.py first to train the initial models and create the state file.")
#         exit() # Exit the script

# def save_state(state):
#     """Saves the current state to the JSON file."""
#     with open(STATE_FILE, "w") as f:
#         json.dump(state, f, indent=4)
#     print(f"DEBUG: State saved. (Processed rows: {state['last_processed_row']})")

# # ------------------------------------------------------------------
# # Data Processing and Prediction Functions
# # ------------------------------------------------------------------

# def preprocess_event(event, models):
#     """Preprocesses a single event dictionary for prediction."""
#     feature_names = models.get("feature_names")
#     if not feature_names:
#         raise ValueError("feature_names not found in loaded models.")

#     x = np.array([event.get(f, np.nan) for f in feature_names], dtype=float).reshape(1, -1)

#     imputer_data = models.get("imputer", {})
#     imputer_map = imputer_data.get("imputer", {})
#     for i, f in enumerate(feature_names):
#         if np.isnan(x[0, i]):
#             x[0, i] = imputer_map.get(f, 0.0)

#     scaler = models.get("scaler")
#     if scaler is not None:
#         x = scaler.transform(x)

#     return x

# def run_models(x, models, buffer):
#     """Runs all loaded models on a preprocessed data point."""
#     results = {}
    
#     # Traditional models predict on the latest event
#     for name in ["isolation_forest", "lof", "elliptic_envelope", "oneclass_svm", "rf_model"]:
#         model = models.get(name)
#         if model is not None:
#             results[name] = int(model.predict(x)[0])

#     # LSTM Autoencoder logic
#     lstm = models.get("lstm_autoencoder")
#     threshold = models.get("lstm_threshold", 0.01)
    
#     if lstm is not None and len(buffer) == 10:
#         try:
#             x_seq = np.array(list(buffer)).reshape(1, 10, -1)
#             recon = lstm.predict(x_seq, verbose=0)
#             loss = np.mean(np.square(x_seq - recon))
#             results["lstm_autoencoder"] = int(loss > threshold)
#             results["lstm_loss"] = float(loss)
#         except Exception as e:
#             results["lstm_autoencoder"] = f"error: {e}"
#     else:
#         results["lstm_autoencoder"] = "pending_buffer"

#     return results

# # ------------------------------------------------------------------
# # Main Monitoring Loop
# # ------------------------------------------------------------------

# def main():
#     """Main function to run the continuous monitoring loop."""
#     # --- Initialization ---
#     print("🚀 Initializing CUA Monitor...")
#     models = load_models(MODEL_DIR)
#     feature_names = models.get("feature_names", [])
#     if not feature_names:
#         print("[ERROR] No feature_names found in models. Exiting.")
#         return
        
#     # --- RCM and Adaptive Learning Parameters ---
#     LAMBDA = 0.90 # Made slightly more responsive
#     LOCKOUT_THRESHOLD = 0.60
#     TRUSTED_THRESHOLD = 0.90
#     RETRAIN_BUFFER_SIZE = 200

#     # --- Weighted Voting Parameters ---
#     MODEL_WEIGHTS = {
#         "lstm_autoencoder": 3,
#         "isolation_forest": 2,
#         "oneclass_svm": 2,
#         "lof": 1,
#         "elliptic_envelope": 1,
#         "rf_model": 1
#     }
#     ANOMALY_THRESHOLD = 3 # Tweak this to adjust sensitivity

#     # --- Buffers ---
#     history_buffer = deque(maxlen=10)
#     trusted_data_buffer = []
    
#     print("✅ Initialization complete. Starting monitoring loop...")

#     # --- Main Loop ---
#     while True:
#         try:
#             state = load_state()
#             confidence_score = state["confidence_score"]
            
#             new_data_df = pd.read_csv(
#                 CD_VECTOR_FILE, 
#                 skiprows=range(1, state["last_processed_row"] + 1)
#             )
            
#             if new_data_df.empty:
#                 print("No new activity detected. Waiting...")
#                 time.sleep(5)
#                 continue

#             print(f"DEBUG: Found {len(new_data_df)} new activity rows to process.")

#             for index, row in new_data_df.iterrows():
#                 absolute_row_index = state['last_processed_row'] + index + 1
#                 event = row.to_dict()
#                 x = preprocess_event(event, models)
#                 history_buffer.append(x[0])
#                 results = run_models(x, models, history_buffer)

#                 # --- RCM Logic (Balanced & Weighted Version) ---
#                 anomaly_weight = 0
#                 vote_details = {}

#                 for model_name, prediction in results.items():
#                     if not isinstance(prediction, int):
#                         continue
                    
#                     is_anomaly_vote = False
#                     if model_name == "lstm_autoencoder" and prediction == 1:
#                         is_anomaly_vote = True
#                     elif model_name != "lstm_autoencoder" and prediction == -1:
#                         is_anomaly_vote = True

#                     vote_details[model_name] = prediction
#                     if is_anomaly_vote:
#                         anomaly_weight += MODEL_WEIGHTS.get(model_name, 1)

#                 is_anomaly = anomaly_weight >= ANOMALY_THRESHOLD
#                 p_t = 0.0 if is_anomaly else 1.0
#                 confidence_score = (LAMBDA * confidence_score) + ((1 - LAMBDA) * p_t)

#                 # Enhanced Debugging Print
#                 vote_str = (
#                     f'IF:{vote_details.get("isolation_forest", "N/A")} '
#                     f'SVM:{vote_details.get("oneclass_svm", "N/A")} '
#                     f'LOF:{vote_details.get("lof", "N/A")} '
#                     f'EE:{vote_details.get("elliptic_envelope", "N/A")} '
#                     f'LSTM:{vote_details.get("lstm_autoencoder", "N/A")} '
#                     f'RF:{vote_details.get("rf_model", "N/A")}'
#                 )
#                 print(
#                     f"Row {absolute_row_index}: "
#                     f"Confidence = {confidence_score:.2f} | "
#                     f"Anomaly = {is_anomaly} | "
#                     f"Weight: {anomaly_weight}/{ANOMALY_THRESHOLD} | "
#                     f"Votes: [{vote_str}]"
#                 )

#                 # Lockout and Buffer Logic
#                 if confidence_score < LOCKOUT_THRESHOLD:
#                     print(f"🚨 ALERT: Intruder detected at row {absolute_row_index}! Confidence dropped to {confidence_score:.2f}.")
#                     trusted_data_buffer.clear()
#                 elif confidence_score > TRUSTED_THRESHOLD:
#                     trusted_data_buffer.append(row.to_dict())

#             # Update and save state after processing the batch
#             state["last_processed_row"] += len(new_data_df)
#             state["confidence_score"] = confidence_score
#             state["trusted_buffer_size"] = len(trusted_data_buffer)
#             save_state(state)

#             # Check for retraining
#             if len(trusted_data_buffer) >= RETRAIN_BUFFER_SIZE:
#                 print("🚀 ADAPTIVE TRAINING: Buffer full. Triggering model retraining...")
#                 # Placeholder for retraining logic
#                 trusted_data_buffer.clear()
                
#         except FileNotFoundError:
#             print(f"ERROR: Cannot find {CD_VECTOR_FILE}. Make sure the capture service is running.")
#             time.sleep(10)
#         except Exception as e:
#             print(f"An unexpected error occurred: {e}")
#             time.sleep(10)

# if __name__ == "__main__":
#     main()

import os
import json
import time
import pickle
import joblib
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from collections import deque
import subprocess # ### NEW IMPORT ###
import sys        # ### NEW IMPORT ###

# --- Constants and File Paths ---
STATE_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cua_state.json"))
CD_VECTOR_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cd_vector.csv"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
TRAIN_SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "train_models.py"))

# ------------------------------------------------------------------
# Model and State Loading Functions
# ------------------------------------------------------------------

def load_models(model_dir):
    """Loads all model artifacts from the specified directory."""
    models = {}
    print(f"DEBUG: Loading models from {model_dir}")
    for fname in os.listdir(model_dir):
        fpath = os.path.join(model_dir, fname)
        key = fname.split(".")[0]
        try:
            if fname.endswith(".pkl"): models[key] = joblib.load(fpath)
            elif fname.endswith(".json"):
                with open(fpath, "r") as f: models[key] = json.load(f)
            elif fname.endswith(".npy"): models[key] = np.load(fpath, allow_pickle=True)
            elif fname.endswith(".keras"): models[key] = keras.models.load_model(fpath)
        except Exception as e:
            print(f"[WARN] Could not load {fname}: {e}")
    print("[INFO] Models loaded:", list(models.keys()))
    return models

def load_state():
    """Loads the state from the JSON file, ensuring all necessary keys exist."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            state.setdefault('anomaly_strike_counter', 0)
            return state
    else:
        print("[ERROR] The state file 'cua_state.json' was not found.")
        print("Please run train_models.py first to create the state file.")
        exit()

def save_state(state):
    """Saves the current state to the JSON file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)
    print(f"DEBUG: State saved. (Processed rows: {state['last_processed_row']})")

# ------------------------------------------------------------------
# Data Processing and Prediction Functions
# ------------------------------------------------------------------

def preprocess_event(event, models):
    """Preprocesses a single event dictionary for prediction."""
    feature_names = models.get("feature_names")
    if not feature_names: raise ValueError("feature_names not found in loaded models.")
    x = np.array([event.get(f, np.nan) for f in feature_names], dtype=float).reshape(1, -1)
    imputer_data = models.get("imputer", {})
    imputer_map = imputer_data.get("imputer", {})
    for i, f in enumerate(feature_names):
        if np.isnan(x[0, i]): x[0, i] = imputer_map.get(f, 0.0)
    scaler = models.get("scaler")
    if scaler is not None: x = scaler.transform(x)
    return x

def run_models(x, models, buffer):
    """Runs all loaded models on a preprocessed data point."""
    results = {}
    for name in ["isolation_forest", "lof", "elliptic_envelope", "oneclass_svm", "rf_model"]:
        model = models.get(name)
        if model is not None: results[name] = int(model.predict(x)[0])
    lstm = models.get("lstm_autoencoder")
    threshold = models.get("lstm_threshold", 0.01)
    if lstm is not None and len(buffer) == 10:
        try:
            x_seq = np.array(list(buffer)).reshape(1, 10, -1)
            recon = lstm.predict(x_seq, verbose=0)
            loss = np.mean(np.square(x_seq - recon))
            results["lstm_autoencoder"] = int(loss > threshold)
        except Exception as e:
            results["lstm_autoencoder"] = f"error: {e}"
    else:
        results["lstm_autoencoder"] = "pending_buffer"
    return results

# ------------------------------------------------------------------
# Main Monitoring Loop
# ------------------------------------------------------------------

def main():
    """Main function to run the continuous monitoring loop."""
    print("🚀 Initializing CUA Monitor...")
    models = load_models(MODEL_DIR)
    
    # --- RCM and Adaptive Learning Parameters ---
    LAMBDA = 0.95 
    LOCKOUT_THRESHOLD = 0.60
    TRUSTED_THRESHOLD = 0.90
    RETRAIN_BUFFER_SIZE = 10

    # --- Weighted Voting Parameters ---
    MODEL_WEIGHTS = {
        "lstm_autoencoder": 3, "isolation_forest": 2, "oneclass_svm": 2,
        "lof": 1, "elliptic_envelope": 1, "rf_model": 1
    }
    ANOMALY_THRESHOLD = 2

    # --- "Three Strikes" System Parameters ---
    STRIKE_THRESHOLD = 3 
    STRIKE_PENALTY = 0.7

    history_buffer = deque(maxlen=10)
    trusted_data_buffer = []
    
    print("✅ Initialization complete. Starting monitoring loop...")

    while True:
        try:
            state = load_state()
            confidence_score = state["confidence_score"]
            anomaly_strike_counter = state["anomaly_strike_counter"]
            
            new_data_df = pd.read_csv(CD_VECTOR_FILE, skiprows=range(1, state["last_processed_row"] + 1))
            
            if new_data_df.empty:
                print("No new activity detected. Waiting...")
                time.sleep(5)
                continue

            print(f"DEBUG: Found {len(new_data_df)} new activity rows to process.")

            for index, row in new_data_df.iterrows():
                absolute_row_index = state['last_processed_row'] + index + 1
                event = row.to_dict()
                x = preprocess_event(event, models)
                history_buffer.append(x[0])
                results = run_models(x, models, history_buffer)

                anomaly_weight = 0
                for model_name, prediction in results.items():
                    if not isinstance(prediction, int): continue
                    is_anomaly_vote = (prediction == 1 if model_name == "lstm_autoencoder" else prediction == -1)
                    if is_anomaly_vote:
                        anomaly_weight += MODEL_WEIGHTS.get(model_name, 1)
                is_anomaly = anomaly_weight >= ANOMALY_THRESHOLD

                if is_anomaly:
                    anomaly_strike_counter += 1
                else:
                    anomaly_strike_counter = max(0, anomaly_strike_counter - 1)
                
                p_t = 0.0 if is_anomaly else 1.0
                confidence_score = (LAMBDA * confidence_score) + ((1 - LAMBDA) * p_t)

                if anomaly_strike_counter >= STRIKE_THRESHOLD:
                    print(f"🚨 STRIKE THRESHOLD MET! Applying penalty.")
                    confidence_score *= STRIKE_PENALTY
                    anomaly_strike_counter = 0
                
                print(
                    f"Row {absolute_row_index}: "
                    f"Confidence = {confidence_score:.2f} | "
                    f"Weight: {anomaly_weight}/{ANOMALY_THRESHOLD} | "
                    f"Strikes: {anomaly_strike_counter}/{STRIKE_THRESHOLD}"
                )

                if confidence_score < LOCKOUT_THRESHOLD:
                    print(f"🚨🚨 INTRUDER ALERT AT ROW {absolute_row_index}! Locking session.")
                    trusted_data_buffer.clear()
                    state = {"last_processed_row": absolute_row_index, "model_version": state["model_version"],
                             "confidence_score": 1.0, "trusted_buffer_size": 0, "anomaly_strike_counter": 0}
                    break
                elif confidence_score > TRUSTED_THRESHOLD:
                    trusted_data_buffer.append(row.to_dict())
            
            state["last_processed_row"] += len(new_data_df)
            state["confidence_score"] = confidence_score
            state["trusted_buffer_size"] = len(trusted_data_buffer)
            state["anomaly_strike_counter"] = anomaly_strike_counter
            save_state(state)

            # --- ### FULLY IMPLEMENTED RETRAINING LOGIC ### ---
            if len(trusted_data_buffer) >= RETRAIN_BUFFER_SIZE:
                print("\n🚀 ADAPTIVE TRAINING: Buffer full. Starting model retraining process...")
                
                # 1. Append the new trusted data to the main CSV file
                df_new = pd.DataFrame(trusted_data_buffer)
                df_new.to_csv(CD_VECTOR_FILE, mode='a', header=False, index=False)
                trusted_data_buffer.clear()
                print(f"✅ {len(df_new)} new trusted rows have been added to {CD_VECTOR_FILE}.")
                
                # 2. Call the training script as a separate process
                print("⏳ Calling train_models.py to update models...")
                try:
                    # sys.executable ensures we use the same Python interpreter
                    subprocess.run([sys.executable, TRAIN_SCRIPT_PATH], check=True)
                    print("✅ Training script completed successfully.")
                    
                    # 3. Gracefully restart the monitor to load the new models
                    print("🔄 Restarting monitor to load new models...")
                    # This exits the current script. The launchd service will automatically restart it.
                    sys.exit(0)
                    
                except subprocess.CalledProcessError:
                    print("❌ ERROR: The training script failed to execute. Check logs for details.")
                except FileNotFoundError:
                    print(f"❌ ERROR: Cannot find the training script at {TRAIN_SCRIPT_PATH}")

        except FileNotFoundError:
            print(f"ERROR: Cannot find {CD_VECTOR_FILE}. Make sure the capture service is running.")
            time.sleep(10)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
# # monitor_online_adaptive.py
# import os
# import json
# import time
# import joblib
# import pandas as pd
# from pathlib import Path
# from sklearn.ensemble import IsolationForest
# from dotenv import load_dotenv

# # ======================
# # Load ENV
# # ======================
# load_dotenv()
# BASE_DIR = os.getenv("BASE_DIR", str(Path(__file__).resolve().parent.parent))
# MODEL_DIR = os.path.join(BASE_DIR, "models")
# DATA_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
# STATE_FILE = os.path.join(BASE_DIR, "cua_state.json")
# print(MODEL_DIR)
# # ======================
# # Load Model
# # ======================
# import os
# import sys

# # Absolute path to models directory

# # Required models
# required_models = [
#     "isolation_forest.pkl",
#     "lof.pkl",
#     "oneclass_svm.pkl",
#     "elliptic_envelope.pkl",
#     "rf_model.pkl",
#     "scaler.pkl",
#     "imputer.pkl",
#     "feature_names.pkl",   # change from json to pkl
#     "training_meta.json"
# ]

# # Check if all required models exist
# missing = [m for m in required_models if not os.path.exists(os.path.join(MODEL_DIR, m))]
# print("DEBUG: Checking models in", MODEL_DIR)
# print("DEBUG: Files available:", os.listdir(MODEL_DIR))
# if missing:
#     print(f"[WARN] Missing required models: {missing}")
#     print("Please run train_models.py first to generate them.")
#     sys.exit(1)
# else:
#     print(f"[INFO] All required models found in {MODEL_DIR}")
    

    
# def load_model():
#     import os, pickle, json
#     from tensorflow.keras.models import load_model as keras_load

#     model_dir = "/Users/mannpatel/Desktop/RMS/Implementation/models"

#     models = {}

#     # Example: Random Forest
#     rf_path = os.path.join(model_dir, "rf_model.pkl")
#     if os.path.exists(rf_path):
#         with open(rf_path, "rb") as f:
#             models["rf"] = pickle.load(f)

#     # Example: Isolation Forest
#     if_path = os.path.join(model_dir, "isolation_forest.pkl")
#     if os.path.exists(if_path):
#         with open(if_path, "rb") as f:
#             models["isolation_forest"] = pickle.load(f)

#     # Example: LSTM Autoencoder
#     lstm_path = os.path.join(model_dir, "lstm_autoencoder.keras")
#     if os.path.exists(lstm_path):
#         models["lstm_autoencoder"] = keras_load(lstm_path)

#     # Meta info (optional)
#     meta_path = os.path.join(model_dir, "training_meta.json")
#     if os.path.exists(meta_path):
#         with open(meta_path, "r") as f:
#             models["meta"] = json.load(f)

#     if not models:
#         return None  # truly nothing found
#     return models

# # ======================
# # Load State
# # ======================
# def load_state():
#     if os.path.exists(STATE_FILE):
#         with open(STATE_FILE, "r") as f:
#             return json.load(f)
#     return {"trained_rows": 0, "retrain_threshold": 200}

# def save_state(state):
#     with open(STATE_FILE, "w") as f:
#         json.dump(state, f, indent=4)

# # ======================
# # Capture user activity (dummy example, replace with real keylog/app usage)
# # ======================
# def capture_user_activity():
#     # Example: collect session features
#     return {
#         "keystroke_speed": 2.1,
#         "mouse_movement": 0.9,
#         "active_app": 3,  # encoded app id
#     }

# # ======================
# # Adaptive Training
# # ======================
# def retrain_model(data):
#     print("[INFO] Retraining model with new data...")
#     model = IsolationForest(
#         n_estimators=150, max_samples="auto", contamination=0.05, random_state=42
#     )
#     model.fit(data)
#     model_path = os.path.join(MODEL_DIR, "cua_model.pkl")
#     joblib.dump(model, model_path)
#     print("[INFO] Model retrained and saved.")
#     return model

# # ======================
# # Main Monitor Loop
# # ======================
# def main():
#     model = load_model()
#     if not model:
#         print("[WARN] No pre-trained model found. Please run train_models.py first.")
#         return

#     state = load_state()
#     print(f"[INFO] Starting CUA monitor. Rows already trained: {state['trained_rows']}")

#     buffer = []  # holds new rows until retrain

#     while True:
#         activity = capture_user_activity()
#         df = pd.DataFrame([activity])

#         # Predict
#         prediction = model.predict(df)[0]  # 1 = owner, -1 = intruder
#         if prediction == 1:
#             print("[OK] Owner detected")
#             buffer.append(activity)
#         else:
#             print("[ALERT] Intruder detected!")

#         # Retrain when buffer large enough
#         if len(buffer) >= state["retrain_threshold"]:
#             df_new = pd.DataFrame(buffer)
#             model = retrain_model(df_new)
#             state["trained_rows"] += len(buffer)
#             save_state(state)
#             buffer = []

#         time.sleep(2)  # poll every 2 sec

# if __name__ == "__main__":
#     main()



# import os
# import json
# import time
# import pickle
# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow.keras as keras
# from collections import deque

# # --- Constants and File Paths ---
# STATE_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cua_state.json"))
# CD_VECTOR_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cd_vector.csv"))
# MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# # ------------------------------------------------------------------
# # Model and State Loading Functions
# # ------------------------------------------------------------------

# def load_models(model_dir):
#     """Loads all model artifacts from the specified directory."""
#     models = {}
#     print(f"DEBUG: Loading models from {model_dir}")
#     for fname in os.listdir(model_dir):
#         fpath = os.path.join(model_dir, fname)
#         key = fname.split(".")[0]

#         try:
#             if fname.endswith(".pkl"):
#                 models[key] = joblib.load(fpath)
#             elif fname.endswith(".json"):
#                 with open(fpath, "r") as f:
#                     models[key] = json.load(f)
#             elif fname.endswith(".npy"):
#                 models[key] = np.load(fpath, allow_pickle=True)
#             elif fname.endswith(".keras"):
#                 models[key] = keras.models.load_model(fpath)
#         except Exception as e:
#             print(f"[WARN] Could not load {fname}: {e}")

#     print("[INFO] Models loaded:", list(models.keys()))
#     return models

# def load_state():
#     """Loads the state from the JSON file. Exits if not found."""
#     if os.path.exists(STATE_FILE):
#         with open(STATE_FILE, "r") as f:
#             return json.load(f)
#     else:
#         # If the state file doesn't exist, we cannot proceed.
#         print("[ERROR] The state file 'cua_state.json' was not found.")
#         print("Please run train_models.py first to train the initial models and create the state file.")
#         exit() # Exit the script

# def save_state(state):
#     """Saves the current state to the JSON file."""
#     with open(STATE_FILE, "w") as f:
#         json.dump(state, f, indent=4)
#     print(f"DEBUG: State saved. (Processed rows: {state['last_processed_row']})")

# # ------------------------------------------------------------------
# # Data Processing and Prediction Functions
# # ------------------------------------------------------------------

# def preprocess_event(event, models):
#     """Preprocesses a single event dictionary for prediction."""
#     feature_names = models.get("feature_names")
#     if not feature_names:
#         raise ValueError("feature_names not found in loaded models.")

#     x = np.array([event.get(f, np.nan) for f in feature_names], dtype=float).reshape(1, -1)

#     imputer_data = models.get("imputer", {})
#     imputer_map = imputer_data.get("imputer", {})
#     for i, f in enumerate(feature_names):
#         if np.isnan(x[0, i]):
#             x[0, i] = imputer_map.get(f, 0.0)

#     scaler = models.get("scaler")
#     if scaler is not None:
#         x = scaler.transform(x)

#     return x

# def run_models(x, models, buffer):
#     """Runs all loaded models on a preprocessed data point."""
#     results = {}
    
#     # Traditional models predict on the latest event
#     for name in ["isolation_forest", "lof", "elliptic_envelope", "oneclass_svm", "rf_model"]:
#         model = models.get(name)
#         if model is not None:
#             results[name] = int(model.predict(x)[0])

#     # LSTM Autoencoder logic
#     lstm = models.get("lstm_autoencoder")
#     threshold = models.get("lstm_threshold", 0.01)
    
#     if lstm is not None and len(buffer) == 10:
#         try:
#             x_seq = np.array(list(buffer)).reshape(1, 10, -1)
#             recon = lstm.predict(x_seq, verbose=0)
#             loss = np.mean(np.square(x_seq - recon))
#             results["lstm_autoencoder"] = int(loss > threshold)
#             results["lstm_loss"] = float(loss)
#         except Exception as e:
#             results["lstm_autoencoder"] = f"error: {e}"
#     else:
#         results["lstm_autoencoder"] = "pending_buffer"

#     return results

# # ------------------------------------------------------------------
# # Main Monitoring Loop
# # ------------------------------------------------------------------

# def main():
#     """Main function to run the continuous monitoring loop."""
#     # --- Initialization ---
#     print("🚀 Initializing CUA Monitor...")
#     models = load_models(MODEL_DIR)
#     feature_names = models.get("feature_names", [])
#     if not feature_names:
#         print("[ERROR] No feature_names found in models. Exiting.")
#         return
        
#     # --- RCM and Adaptive Learning Parameters ---
#     LAMBDA = 0.90 # Made slightly more responsive
#     LOCKOUT_THRESHOLD = 0.60
#     TRUSTED_THRESHOLD = 0.90
#     RETRAIN_BUFFER_SIZE = 200

#     # --- Weighted Voting Parameters ---
#     MODEL_WEIGHTS = {
#         "lstm_autoencoder": 3,
#         "isolation_forest": 2,
#         "oneclass_svm": 2,
#         "lof": 1,
#         "elliptic_envelope": 1,
#         "rf_model": 1
#     }
#     ANOMALY_THRESHOLD = 3 # Tweak this to adjust sensitivity

#     # --- Buffers ---
#     history_buffer = deque(maxlen=10)
#     trusted_data_buffer = []
    
#     print("✅ Initialization complete. Starting monitoring loop...")

#     # --- Main Loop ---
#     while True:
#         try:
#             state = load_state()
#             confidence_score = state["confidence_score"]
            
#             new_data_df = pd.read_csv(
#                 CD_VECTOR_FILE, 
#                 skiprows=range(1, state["last_processed_row"] + 1)
#             )
            
#             if new_data_df.empty:
#                 print("No new activity detected. Waiting...")
#                 time.sleep(5)
#                 continue

#             print(f"DEBUG: Found {len(new_data_df)} new activity rows to process.")

#             for index, row in new_data_df.iterrows():
#                 absolute_row_index = state['last_processed_row'] + index + 1
#                 event = row.to_dict()
#                 x = preprocess_event(event, models)
#                 history_buffer.append(x[0])
#                 results = run_models(x, models, history_buffer)

#                 # --- RCM Logic (Balanced & Weighted Version) ---
#                 anomaly_weight = 0
#                 vote_details = {}

#                 for model_name, prediction in results.items():
#                     if not isinstance(prediction, int):
#                         continue
                    
#                     is_anomaly_vote = False
#                     if model_name == "lstm_autoencoder" and prediction == 1:
#                         is_anomaly_vote = True
#                     elif model_name != "lstm_autoencoder" and prediction == -1:
#                         is_anomaly_vote = True

#                     vote_details[model_name] = prediction
#                     if is_anomaly_vote:
#                         anomaly_weight += MODEL_WEIGHTS.get(model_name, 1)

#                 is_anomaly = anomaly_weight >= ANOMALY_THRESHOLD
#                 p_t = 0.0 if is_anomaly else 1.0
#                 confidence_score = (LAMBDA * confidence_score) + ((1 - LAMBDA) * p_t)

#                 # Enhanced Debugging Print
#                 vote_str = (
#                     f'IF:{vote_details.get("isolation_forest", "N/A")} '
#                     f'SVM:{vote_details.get("oneclass_svm", "N/A")} '
#                     f'LOF:{vote_details.get("lof", "N/A")} '
#                     f'EE:{vote_details.get("elliptic_envelope", "N/A")} '
#                     f'LSTM:{vote_details.get("lstm_autoencoder", "N/A")} '
#                     f'RF:{vote_details.get("rf_model", "N/A")}'
#                 )
#                 print(
#                     f"Row {absolute_row_index}: "
#                     f"Confidence = {confidence_score:.2f} | "
#                     f"Anomaly = {is_anomaly} | "
#                     f"Weight: {anomaly_weight}/{ANOMALY_THRESHOLD} | "
#                     f"Votes: [{vote_str}]"
#                 )

#                 # Lockout and Buffer Logic
#                 if confidence_score < LOCKOUT_THRESHOLD:
#                     print(f"🚨 ALERT: Intruder detected at row {absolute_row_index}! Confidence dropped to {confidence_score:.2f}.")
#                     trusted_data_buffer.clear()
#                 elif confidence_score > TRUSTED_THRESHOLD:
#                     trusted_data_buffer.append(row.to_dict())

#             # Update and save state after processing the batch
#             state["last_processed_row"] += len(new_data_df)
#             state["confidence_score"] = confidence_score
#             state["trusted_buffer_size"] = len(trusted_data_buffer)
#             save_state(state)

#             # Check for retraining
#             if len(trusted_data_buffer) >= RETRAIN_BUFFER_SIZE:
#                 print("🚀 ADAPTIVE TRAINING: Buffer full. Triggering model retraining...")
#                 # Placeholder for retraining logic
#                 trusted_data_buffer.clear()
                
#         except FileNotFoundError:
#             print(f"ERROR: Cannot find {CD_VECTOR_FILE}. Make sure the capture service is running.")
#             time.sleep(10)
#         except Exception as e:
#             print(f"An unexpected error occurred: {e}")
#             time.sleep(10)

# if __name__ == "__main__":
#     main()

import os
import json
import time
import pickle
import joblib
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from collections import deque
import subprocess # ### NEW IMPORT ###
import sys        # ### NEW IMPORT ###

# --- Constants and File Paths ---
STATE_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cua_state.json"))
CD_VECTOR_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cd_vector.csv"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
TRAIN_SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "train_models.py"))

# ------------------------------------------------------------------
# Model and State Loading Functions
# ------------------------------------------------------------------

def load_models(model_dir):
    """Loads all model artifacts from the specified directory."""
    models = {}
    print(f"DEBUG: Loading models from {model_dir}")
    for fname in os.listdir(model_dir):
        fpath = os.path.join(model_dir, fname)
        key = fname.split(".")[0]
        try:
            if fname.endswith(".pkl"): models[key] = joblib.load(fpath)
            elif fname.endswith(".json"):
                with open(fpath, "r") as f: models[key] = json.load(f)
            elif fname.endswith(".npy"): models[key] = np.load(fpath, allow_pickle=True)
            elif fname.endswith(".keras"): models[key] = keras.models.load_model(fpath)
        except Exception as e:
            print(f"[WARN] Could not load {fname}: {e}")
    print("[INFO] Models loaded:", list(models.keys()))
    return models

def load_state():
    """Loads the state from the JSON file, ensuring all necessary keys exist."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            state.setdefault('anomaly_strike_counter', 0)
            return state
    else:
        print("[ERROR] The state file 'cua_state.json' was not found.")
        print("Please run train_models.py first to create the state file.")
        exit()

def save_state(state):
    """Saves the current state to the JSON file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)
    print(f"DEBUG: State saved. (Processed rows: {state['last_processed_row']})")

# ------------------------------------------------------------------
# Data Processing and Prediction Functions
# ------------------------------------------------------------------

def preprocess_event(event, models):
    """Preprocesses a single event dictionary for prediction."""
    feature_names = models.get("feature_names")
    if not feature_names: raise ValueError("feature_names not found in loaded models.")
    x = np.array([event.get(f, np.nan) for f in feature_names], dtype=float).reshape(1, -1)
    imputer_data = models.get("imputer", {})
    imputer_map = imputer_data.get("imputer", {})
    for i, f in enumerate(feature_names):
        if np.isnan(x[0, i]): x[0, i] = imputer_map.get(f, 0.0)
    scaler = models.get("scaler")
    if scaler is not None: x = scaler.transform(x)
    return x

def run_models(x, models, buffer):
    """Runs all loaded models on a preprocessed data point."""
    results = {}
    for name in ["isolation_forest", "lof", "elliptic_envelope", "oneclass_svm", "rf_model"]:
        model = models.get(name)
        if model is not None: results[name] = int(model.predict(x)[0])
    lstm = models.get("lstm_autoencoder")
    threshold = models.get("lstm_threshold", 0.01)
    if lstm is not None and len(buffer) == 10:
        try:
            x_seq = np.array(list(buffer)).reshape(1, 10, -1)
            recon = lstm.predict(x_seq, verbose=0)
            loss = np.mean(np.square(x_seq - recon))
            results["lstm_autoencoder"] = int(loss > threshold)
        except Exception as e:
            results["lstm_autoencoder"] = f"error: {e}"
    else:
        results["lstm_autoencoder"] = "pending_buffer"
    return results

# ------------------------------------------------------------------
# Main Monitoring Loop
# ------------------------------------------------------------------

def main():
    """Main function to run the continuous monitoring loop."""
    print("🚀 Initializing CUA Monitor...")
    models = load_models(MODEL_DIR)
    
    # --- RCM and Adaptive Learning Parameters ---
    LAMBDA = 0.95 
    LOCKOUT_THRESHOLD = 0.60
    TRUSTED_THRESHOLD = 0.90
    RETRAIN_BUFFER_SIZE = 10

    # --- Weighted Voting Parameters ---
    MODEL_WEIGHTS = {
        "lstm_autoencoder": 3, "isolation_forest": 2, "oneclass_svm": 2,
        "lof": 1, "elliptic_envelope": 1, "rf_model": 1
    }
    ANOMALY_THRESHOLD = 2

    # --- "Three Strikes" System Parameters ---
    STRIKE_THRESHOLD = 3 
    STRIKE_PENALTY = 0.7

    history_buffer = deque(maxlen=10)
    trusted_data_buffer = []
    
    print("✅ Initialization complete. Starting monitoring loop...")

    while True:
        try:
            state = load_state()
            confidence_score = state["confidence_score"]
            anomaly_strike_counter = state["anomaly_strike_counter"]
            
            new_data_df = pd.read_csv(CD_VECTOR_FILE, skiprows=range(1, state["last_processed_row"] + 1))
            
            if new_data_df.empty:
                print("No new activity detected. Waiting...")
                time.sleep(5)
                continue

            print(f"DEBUG: Found {len(new_data_df)} new activity rows to process.")

            for index, row in new_data_df.iterrows():
                absolute_row_index = state['last_processed_row'] + index + 1
                event = row.to_dict()
                x = preprocess_event(event, models)
                history_buffer.append(x[0])
                results = run_models(x, models, history_buffer)

                anomaly_weight = 0
                for model_name, prediction in results.items():
                    if not isinstance(prediction, int): continue
                    is_anomaly_vote = (prediction == 1 if model_name == "lstm_autoencoder" else prediction == -1)
                    if is_anomaly_vote:
                        anomaly_weight += MODEL_WEIGHTS.get(model_name, 1)
                is_anomaly = anomaly_weight >= ANOMALY_THRESHOLD

                if is_anomaly:
                    anomaly_strike_counter += 1
                else:
                    anomaly_strike_counter = max(0, anomaly_strike_counter - 1)
                
                p_t = 0.0 if is_anomaly else 1.0
                confidence_score = (LAMBDA * confidence_score) + ((1 - LAMBDA) * p_t)

                if anomaly_strike_counter >= STRIKE_THRESHOLD:
                    print(f"🚨 STRIKE THRESHOLD MET! Applying penalty.")
                    confidence_score *= STRIKE_PENALTY
                    anomaly_strike_counter = 0
                
                print(
                    f"Row {absolute_row_index}: "
                    f"Confidence = {confidence_score:.2f} | "
                    f"Weight: {anomaly_weight}/{ANOMALY_THRESHOLD} | "
                    f"Strikes: {anomaly_strike_counter}/{STRIKE_THRESHOLD}"
                )

                if confidence_score < LOCKOUT_THRESHOLD:
                    print(f"🚨🚨 INTRUDER ALERT AT ROW {absolute_row_index}! Locking session.")
                    trusted_data_buffer.clear()
                    state = {"last_processed_row": absolute_row_index, "model_version": state["model_version"],
                             "confidence_score": 1.0, "trusted_buffer_size": 0, "anomaly_strike_counter": 0}
                    break
                elif confidence_score > TRUSTED_THRESHOLD:
                    trusted_data_buffer.append(row.to_dict())
            
            state["last_processed_row"] += len(new_data_df)
            state["confidence_score"] = confidence_score
            state["trusted_buffer_size"] = len(trusted_data_buffer)
            state["anomaly_strike_counter"] = anomaly_strike_counter
            save_state(state)

            # --- ### FULLY IMPLEMENTED RETRAINING LOGIC ### ---
            if len(trusted_data_buffer) >= RETRAIN_BUFFER_SIZE:
                print("\n🚀 ADAPTIVE TRAINING: Buffer full. Starting model retraining process...")
                
                # 1. Append the new trusted data to the main CSV file
                df_new = pd.DataFrame(trusted_data_buffer)
                df_new.to_csv(CD_VECTOR_FILE, mode='a', header=False, index=False)
                trusted_data_buffer.clear()
                print(f"✅ {len(df_new)} new trusted rows have been added to {CD_VECTOR_FILE}.")
                
                # 2. Call the training script as a separate process
                print("⏳ Calling train_models.py to update models...")
                try:
                    # sys.executable ensures we use the same Python interpreter
                    subprocess.run([sys.executable, TRAIN_SCRIPT_PATH], check=True)
                    print("✅ Training script completed successfully.")
                    
                    # 3. Gracefully restart the monitor to load the new models
                    print("🔄 Restarting monitor to load new models...")
                    # This exits the current script. The launchd service will automatically restart it.
                    sys.exit(0)
                    
                except subprocess.CalledProcessError:
                    print("❌ ERROR: The training script failed to execute. Check logs for details.")
                except FileNotFoundError:
                    print(f"❌ ERROR: Cannot find the training script at {TRAIN_SCRIPT_PATH}")

        except FileNotFoundError:
            print(f"ERROR: Cannot find {CD_VECTOR_FILE}. Make sure the capture service is running.")
            time.sleep(10)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
