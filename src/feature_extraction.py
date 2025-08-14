# # feature_extraction.py

# import pandas as pd
# import json
# import os

# OFFSET_PATH = "/Users/mannpatel/Desktop/RMS/Implementation/data/offset.json"


# def load_offsets():
#     if os.path.exists(OFFSET_PATH):
#         with open(OFFSET_PATH, "r") as f:
#             return json.load(f)
#     return {"keystroke": 0, "mouse_move": 0, "mouse_click": 0}

# def save_offsets(offsets):
#     with open(OFFSET_PATH, "w") as f:
#         json.dump(offsets, f)

# def generate_cd_from_new_data(label=1, min_rows=30):
#     offsets = load_offsets()

#     ks_path = "/Users/mannpatel/Desktop/RMS/Implementation/data/keystroke_features.csv"
#     mm_path = "/Users/mannpatel/Desktop/RMS/Implementation/data/mouse_motion_features.csv"
#     mc_path = "/Users/mannpatel/Desktop/RMS/Implementation/data/mouse_click_features.csv"


#     keystrokes_df = pd.read_csv(ks_path)
#     mouse_move_df = pd.read_csv(mm_path)
#     mouse_click_df = pd.read_csv(mc_path)

#     new_ks = keystrokes_df[offsets["keystroke"]:]
#     new_mm = mouse_move_df[offsets["mouse_move"]:]
#     new_mc = mouse_click_df[offsets["mouse_click"]:]

#     if len(new_ks) >= min_rows and len(new_mm) >= min_rows and len(new_mc) >= min_rows:
#         from generate_cd_vector import generate_cd_vector
#         cd_vector = generate_cd_vector(new_ks, new_mm, new_mc, label)

#         # Update offsets
#         offsets["keystroke"] += len(new_ks)
#         offsets["mouse_move"] += len(new_mm)
#         offsets["mouse_click"] += len(new_mc)
#         save_offsets(offsets)

#         return cd_vector

#     return None

import pandas as pd
import json
import os
import datetime
from dotenv import load_dotenv

load_dotenv()

base_dir = os.getenv('base_dir')
BASE_DIR = f"{base_dir}/data/"
# === Configurable Paths ===

OFFSET_PATH = os.path.join(BASE_DIR, "offset.json")
KS_PATH = os.path.join(BASE_DIR, "keystroke_features.csv")
MM_PATH = os.path.join(BASE_DIR, "mouse_motion_features.csv")
MC_PATH = os.path.join(BASE_DIR, "mouse_click_features.csv")

# === Load Offset Values ===
def load_offsets():
    if os.path.exists(OFFSET_PATH):
        with open(OFFSET_PATH, "r") as f:
            return json.load(f)
    return {"keystroke": 0, "mouse_move": 0, "mouse_click": 0}

# === Save Updated Offsets ===
def save_offsets(offsets):
    with open(OFFSET_PATH, "w") as f:
        json.dump(offsets, f)

# === Main Function to Generate CD Vector from New Data ===
def generate_cd_from_new_data(label=1):
    offsets = load_offsets()

    keystrokes_df = pd.read_csv(KS_PATH)
    mouse_move_df = pd.read_csv(MM_PATH)
    mouse_click_df = pd.read_csv(MC_PATH)

    new_ks = keystrokes_df[offsets["keystroke"]:]
    new_mm = mouse_move_df[offsets["mouse_move"]:]
    new_mc = mouse_click_df[offsets["mouse_click"]:]

    # Skip if data is insufficient or contains incomplete rows
    total_activity = len(new_ks) + len(new_mm) + len(new_mc)

    if (
        total_activity < 20  # <-- global threshold check here
        or (len(new_ks) < 4 and len(new_mm) < 10 and len(new_mc) < 2)
        or new_ks.isnull().values.any()
        or new_mm.isnull().values.any()
        or new_mc.isnull().values.any()
    ):
        return None


    # Generate CD vector
    from generate_cd_vector import generate_cd_vector
    cd_vector = generate_cd_vector(new_ks, new_mm, new_mc, label)

    # Add timestamp
    cd_vector["Timestamp"] = datetime.datetime.now().isoformat()

    # Update offsets
    offsets["keystroke"] += len(new_ks)
    offsets["mouse_move"] += len(new_mm)
    offsets["mouse_click"] += len(new_mc)
    save_offsets(offsets)

    return cd_vector
