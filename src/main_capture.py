# import time
# import os
# import pandas as pd
# from datetime import datetime
# from feature_extraction import generate_cd_from_new_data
# from dotenv import load_dotenv

# load_dotenv()

# base_dir = os.getenv('base_dir')
# csv_path = f"{base_dir}/data/cd_vector.csv"


# THRESHOLD = 20  # Minimum combined activity count to record a CD vector

# HEADERS = [
#     "MeanHoldTime", "StdHoldTime",
#     "MeanSpeed", "StdSpeed",
#     "MeanAcceleration", "StdAcceleration",
#     "MeanClickDuration", "StdClickDuration",
#     "KeystrokeCount", "MouseMoveCount", "MouseClickCount",
#     "Label", "Timestamp"
# ]

# # Initialize CSV with headers
# if not os.path.isfile(csv_path):
#     pd.DataFrame(columns=HEADERS).to_csv(csv_path, index=False)
#     print("✅ CSV initialized with headers.")
# else:
#     try:
#         df = pd.read_csv(csv_path)
#         if list(df.columns) != HEADERS:
#             print("⚠️ CSV headers incorrect. Fixing...")
#             df.columns = HEADERS
#             df.to_csv(csv_path, index=False)
#     except Exception as e:
#         print(f"⚠️ Error reading CSV: {e}. Recreating...")
#         pd.DataFrame(columns=HEADERS).to_csv(csv_path, index=False)

# print("🚀 Starting data collection...")

# while True:
#     try:
#         cd_vector = generate_cd_from_new_data(label=1)
#         if cd_vector is not None and not cd_vector.empty:
#             total_activity = (
#                 cd_vector["KeystrokeCount"].iloc[0] +
#                 cd_vector["MouseMoveCount"].iloc[0] +
#                 cd_vector["MouseClickCount"].iloc[0]
#             )

#             if total_activity >= THRESHOLD:
#                 cd_vector["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 cd_vector = cd_vector[HEADERS]  # ensure correct column order
#                 cd_vector.to_csv(csv_path, mode='a', index=False, header=False)
#                 print("✅ CD vector saved.")
#             else:
#                 print(f"⏳ Waiting for more activity... (Current: {total_activity})")
#         else:
#             print("⚠️ No data to save.")
#     except Exception as e:
#         print(f"❌ Error generating/saving CD vector: {e}")

#     time.sleep(5)

import time
import os
import pandas as pd
from datetime import datetime
from feature_extraction import generate_cd_from_new_data
from dotenv import load_dotenv

load_dotenv()

base_dir = os.getenv('base_dir')
csv_path = f"{base_dir}/data/cd_vector.csv"


THRESHOLD = 20  # Minimum combined activity count to record a CD vector

HEADERS = [
    "MeanHoldTime", "StdHoldTime",
    "MeanSpeed", "StdSpeed",
    "MeanAcceleration", "StdAcceleration",
    "MeanClickDuration", "StdClickDuration",
    "KeystrokeCount", "MouseMoveCount", "MouseClickCount",
    "Label", "Timestamp"
]

# --- SAFER CSV Initialization ---
# Check if the file exists and is not empty.
# os.path.getsize() checks the file size in bytes.
file_exists_and_is_not_empty = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0

if not file_exists_and_is_not_empty:
    # If the file doesn't exist or is empty, create it with headers.
    try:
        pd.DataFrame(columns=HEADERS).to_csv(csv_path, index=False)
        print("✅ CSV file was missing or empty. Initialized with headers.")
    except Exception as e:
        print(f"❌ Critical error: Could not create or write to CSV file at {csv_path}. Error: {e}")
        # In a real-world scenario, you might want to exit here if you can't write data.

print("🚀 Starting data collection...")

while True:
    try:
        cd_vector = generate_cd_from_new_data(label=1)
        if cd_vector is not None and not cd_vector.empty:
            total_activity = (
                cd_vector["KeystrokeCount"].iloc[0] +
                cd_vector["MouseMoveCount"].iloc[0] +
                cd_vector["MouseClickCount"].iloc[0]
            )

            if total_activity >= THRESHOLD:
                cd_vector["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cd_vector = cd_vector[HEADERS]  # ensure correct column order
                cd_vector.to_csv(csv_path, mode='a', index=False, header=False)
                print("✅ CD vector saved.")
            else:
                print(f"⏳ Waiting for more activity... (Current: {total_activity})")
        else:
            print("⚠️ No data to save.")
    except Exception as e:
        print(f"❌ Error generating/saving CD vector: {e}")

    time.sleep(5)