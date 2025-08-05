# #!/usr/bin/env python
# # coding: utf-8

# from pynput import keyboard, mouse
# import time, csv, os, signal, sys
# from datetime import datetime
# import logging
# import pandas as pd
# import numpy as np

# # === Ensure data folder exists ===
# data_dir = "/Users/mannpatel/Desktop/RMS/Implementation/data"
# os.makedirs(data_dir, exist_ok=True)

# # === Setup Logging ===
# log_file = os.path.join(data_dir, "cua.log")
# logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
# logging.info("Starting CUA data capture and feature extraction service.")

# # === File paths (all in data directory) ===
# keyboard_file = os.path.join(data_dir, "keyboard_raw.csv")
# mouse_file = os.path.join(data_dir, "mouse_raw.csv")
# keystroke_features_file = os.path.join(data_dir, "keystroke_features.csv")
# mouse_motion_features_file = os.path.join(data_dir, "mouse_motion_features.csv")
# mouse_click_features_file = os.path.join(data_dir, "mouse_click_features.csv")

# # === Open raw data files in append mode ===
# kb_file = open(keyboard_file, "a", newline="")
# ms_file = open(mouse_file, "a", newline="")

# kb_writer = csv.writer(kb_file)
# ms_writer = csv.writer(ms_file)

# # === Initialize feature files with headers if empty ===
# for file_path, headers in [
#     (keystroke_features_file, ["HoldTime"]),
#     (mouse_motion_features_file, ["Speed", "Acceleration"]),
#     (mouse_click_features_file, ["ClickDuration"])
# ]:
#     if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
#         with open(file_path, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(headers)

# # === Write initial raw data headers if empty ===
# if os.stat(keyboard_file).st_size == 0:
#     kb_writer.writerow(["event_type", "timestamp", "datetime"])

# if os.stat(mouse_file).st_size == 0:
#     ms_writer.writerow(["event_type", "x", "y", "timestamp", "datetime"])

# # === Mouse listener needs to be globally accessible ===
# ms_listener = None

# # === Feature Extraction Functions ===
# def extract_keystroke_features(df):
#     features = []
#     press_time = None
#     for _, row in df.iterrows():
#         event, t = row["event_type"], float(row["timestamp"])
#         if event == "press":
#             press_time = t
#         elif event == "release" and press_time is not None:
#             hold_time = t - press_time
#             features.append({"HoldTime": hold_time})
#             press_time = None
#     return pd.DataFrame(features)

# def extract_mouse_features(df):
#     move_features = []
#     click_features = []
#     prev = {"x": None, "y": None, "t": None, "speed": None}
#     click_start_time = None
#     for _, row in df.iterrows():
#         event, t = row["event_type"], float(row["timestamp"])
#         if event == "move":
#             x, y = float(row["x"]), float(row["y"])
#             if prev["x"] is not None:
#                 dist = ((x - prev["x"])**2 + (y - prev["y"])**2)**0.5
#                 dt = t - prev["t"] + 1e-6
#                 speed = dist / dt
#                 acc = (speed - prev["speed"]) / dt if prev["speed"] is not None else 0
#                 move_features.append({"Speed": speed, "Acceleration": acc})
#                 prev["speed"] = speed
#             prev.update({"x": x, "y": y, "t": t})
#         elif event == "press":
#             click_start_time = t
#         elif event == "release" and click_start_time is not None:
#             duration = t - click_start_time
#             click_features.append({"ClickDuration": duration})
#             click_start_time = None
#     return pd.DataFrame(move_features), pd.DataFrame(click_features)

# # === Keyboard Handlers ===
# def on_press(key):
#     kb_writer.writerow(["press", time.time(), datetime.now()])
#     kb_file.flush()
#     logging.info("Key press detected")

# def on_release(key):
#     kb_writer.writerow(["release", time.time(), datetime.now()])
#     kb_file.flush()
#     logging.info("Key release detected")
#     if key == keyboard.Key.esc:
#         stop_listeners()

# # === Mouse Handlers ===
# def on_move(x, y):
#     try:
#         ms_writer.writerow(["move", x, y, time.time(), datetime.now()])
#         ms_file.flush()
#         logging.info(f"Mouse move: ({x}, {y})")
#     except ValueError:
#         pass

# def on_click(x, y, button, pressed):
#     try:
#         event = "press" if pressed else "release"
#         ms_writer.writerow([event, x, y, time.time(), datetime.now()])
#         ms_file.flush()
#         logging.info(f"Mouse {event}: ({x}, {y})")
#     except ValueError:
#         pass

# # === Cleanup Handler ===
# def stop_listeners(signum=None, frame=None):
#     logging.info("Stopping CUA data capture and feature extraction service.")
#     if ms_listener: ms_listener.stop()
#     kb_file.close()
#     ms_file.close()
#     sys.exit(0)

# # === Signal handler for Ctrl+C and launchctl stop ===
# signal.signal(signal.SIGINT, stop_listeners)
# signal.signal(signal.SIGTERM, stop_listeners)

# # === Start Listeners and Run Feature Extraction ===
# print("🔴 CUA Capture started — Press ESC or Ctrl+C to stop. Move/click mouse to record events.")
# kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
# ms_listener = mouse.Listener(on_move=on_move, on_click=on_click)

# kb_listener.start()
# ms_listener.start()

# # === Continuous Feature Extraction Loop ===
# last_kb_line = sum(1 for _ in open(keyboard_file, 'r')) - 1 if os.path.exists(keyboard_file) and os.stat(keyboard_file).st_size > 0 else 0
# last_ms_line = sum(1 for _ in open(mouse_file, 'r')) - 1 if os.path.exists(mouse_file) and os.stat(mouse_file).st_size > 0 else 0
# while True:
#     try:
#         # Check for new keyboard data
#         with open(keyboard_file, 'r') as f:
#             lines = f.readlines()
#             if len(lines) > last_kb_line + 1:  # +1 to account for header
#                 df_kb = pd.read_csv(keyboard_file, skiprows=range(last_kb_line + 1), 
#                                   names=["event_type", "timestamp", "datetime"])
#                 if not df_kb.empty:
#                     kf = extract_keystroke_features(df_kb)
#                     if not kf.empty:
#                         kf.to_csv(keystroke_features_file, mode='a', header=False, index=False)
#                 last_kb_line = len(lines) - 1  # Update to last data line

#         # Check for new mouse data
#         with open(mouse_file, 'r') as f:
#             lines = f.readlines()
#             if len(lines) > last_ms_line + 1:  # +1 to account for header
#                 df_ms = pd.read_csv(mouse_file, skiprows=range(last_ms_line + 1), 
#                                   names=["event_type", "x", "y", "timestamp", "datetime"])
#                 if not df_ms.empty:
#                     mf_speed_accel, mf_click = extract_mouse_features(df_ms)
#                     if not mf_speed_accel.empty:
#                         mf_speed_accel.to_csv(mouse_motion_features_file, mode='a', header=False, index=False)
#                     if not mf_click.empty:
#                         mf_click.to_csv(mouse_click_features_file, mode='a', header=False, index=False)
#                 last_ms_line = len(lines) - 1  # Update to last data line

#         time.sleep(1)  # Check every second

#     except Exception as e:
#         logging.error(f"Feature extraction error: {e}")
#         time.sleep(5)  # Wait before retrying on error


#!/usr/bin/env python
# coding: utf-8

from pynput import keyboard, mouse
import time, csv, os, signal, sys
from datetime import datetime
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

base_dir = os.getenv('base_dir')
data_dir = f"{base_dir}/data/"
# === File Setup ===

os.makedirs(data_dir, exist_ok=True)

keyboard_file = os.path.join(data_dir, "keyboard_raw.csv")
mouse_file = os.path.join(data_dir, "mouse_raw.csv")
keystroke_features_file = os.path.join(data_dir, "keystroke_features.csv")
mouse_motion_features_file = os.path.join(data_dir, "mouse_motion_features.csv")
mouse_click_features_file = os.path.join(data_dir, "mouse_click_features.csv")

# === Logging ===
log_file = os.path.join(data_dir, "cua.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("CUA capture started.")

# === Init Raw Data Files ===
kb_file = open(keyboard_file, "a", newline="")
ms_file = open(mouse_file, "a", newline="")
kb_writer = csv.writer(kb_file)
ms_writer = csv.writer(ms_file)

if os.stat(keyboard_file).st_size == 0:
    kb_writer.writerow(["event_type", "timestamp", "datetime"])
if os.stat(mouse_file).st_size == 0:
    ms_writer.writerow(["event_type", "x", "y", "timestamp", "datetime"])

# === Init Feature Files ===
for path, headers in [
    (keystroke_features_file, ["HoldTime"]),
    (mouse_motion_features_file, ["Speed", "Acceleration"]),
    (mouse_click_features_file, ["ClickDuration"])
]:
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

# === Feature Extraction ===
def extract_keystroke_features(df):
    features = []
    press_time = None
    for _, row in df.iterrows():
        event, t = row["event_type"], float(row["timestamp"])
        if event == "press":
            press_time = t
        elif event == "release" and press_time is not None:
            features.append({"HoldTime": t - press_time})
            press_time = None
    return pd.DataFrame(features)

def extract_mouse_features(df):
    move_features, click_features = [], []
    prev = {"x": None, "y": None, "t": None, "speed": None}
    click_start = None
    for _, row in df.iterrows():
        event, t = row["event_type"], float(row["timestamp"])
        if event == "move":
            x, y = float(row["x"]), float(row["y"])
            if prev["x"] is not None:
                dist = ((x - prev["x"])**2 + (y - prev["y"])**2)**0.5
                dt = t - prev["t"] + 1e-6
                speed = dist / dt
                acc = (speed - prev["speed"]) / dt if prev["speed"] is not None else 0
                move_features.append({"Speed": speed, "Acceleration": acc})
                prev["speed"] = speed
            prev.update({"x": x, "y": y, "t": t})
        elif event == "press":
            click_start = t
        elif event == "release" and click_start is not None:
            click_features.append({"ClickDuration": t - click_start})
            click_start = None
    return pd.DataFrame(move_features), pd.DataFrame(click_features)

# === Keyboard & Mouse Handlers ===
def on_press(key):
    kb_writer.writerow(["press", time.time(), datetime.now()])
    kb_file.flush()

def on_release(key):
    kb_writer.writerow(["release", time.time(), datetime.now()])
    kb_file.flush()
    # if key == keyboard.Key.esc:
    #     stop()

def on_move(x, y):
    try:
        ms_writer.writerow(["move", x, y, time.time(), datetime.now()])
        ms_file.flush()
    except Exception:
        pass

def on_click(x, y, button, pressed):
    try:
        ms_writer.writerow(["press" if pressed else "release", x, y, time.time(), datetime.now()])
        ms_file.flush()
    except Exception:
        pass

# === Stop & Cleanup ===
def stop(signum=None, frame=None):
    logging.info("Stopping CUA...")
    if ms_listener: ms_listener.stop()
    kb_file.close()
    ms_file.close()
    sys.exit(0)

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

# === Start Listeners ===
# print("🔴 CUA Running — Press ESC or Ctrl+C to stop.")
kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
ms_listener = mouse.Listener(on_move=on_move, on_click=on_click)
kb_listener.start()
ms_listener.start()

# === Live Feature Extraction Loop ===
last_kb_line = sum(1 for _ in open(keyboard_file, 'r')) - 1
last_ms_line = sum(1 for _ in open(mouse_file, 'r')) - 1

while True:
    try:
        # Keyboard
        with open(keyboard_file, 'r') as f:
            lines = f.readlines()
        if len(lines) > last_kb_line + 1:
            df_kb = pd.read_csv(keyboard_file, skiprows=range(0, last_kb_line + 1),
                    header=None, names=["event_type", "timestamp", "datetime"])
            kf = extract_keystroke_features(df_kb)
            if not kf.empty:
                kf.to_csv(keystroke_features_file, mode='a', header=False, index=False)
            last_kb_line = len(lines) - 1

        # Mouse
        with open(mouse_file, 'r') as f:
            lines = f.readlines()
        if len(lines) > last_ms_line + 1:
            df_ms = pd.read_csv(mouse_file, skiprows=range(0, last_ms_line + 1),
                    header=None, names=["event_type", "x", "y", "timestamp", "datetime"])

            mf_move, mf_click = extract_mouse_features(df_ms)
            if not mf_move.empty:
                mf_move.to_csv(mouse_motion_features_file, mode='a', header=False, index=False)
            if not mf_click.empty:
                mf_click.to_csv(mouse_click_features_file, mode='a', header=False, index=False)
            last_ms_line = len(lines) - 1

        time.sleep(1)

    except Exception as e:
        logging.error(f"Error: {e}")
        time.sleep(5)


