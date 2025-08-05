import time
import os
import pandas as pd
from datetime import datetime
from feature_extraction import generate_cd_from_new_data
from dotenv import load_dotenv

load_dotenv()

base_dir = os.getenv('base_dir')
csv_path = f"{base_dir}/data/cd_vector.csv"
print(csv_path)

# THRESHOLD = 50  # Minimum combined activity count to record a CD vector

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
