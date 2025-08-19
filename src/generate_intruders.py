import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv("base_dir")
DATA_DIR = os.path.join(base_dir, "data")

INPUT_CSV = os.path.join(DATA_DIR, "cd_vectors_clean.csv")   # original legit data
OUTPUT_CSV = os.path.join(DATA_DIR, "cd_vector_with_intruders.csv")

def generate_intruders():
    df = pd.read_csv(INPUT_CSV)

    # Drop non-numeric cols (like Timestamp/Label) for augmentation
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # --- Intruder 1: Noise injection ---
    noisy = numeric_df.copy()
    noisy += np.random.normal(loc=0, scale=2, size=noisy.shape)

    # --- Intruder 2: Extreme outliers ---
    outliers = numeric_df.copy()
    for col in outliers.columns:
        outliers[col] = outliers[col] * np.random.choice([0.1, 5, 10], size=len(outliers))

    # --- Intruder 3: Random shuffling ---
    shuffled = numeric_df.copy()
    for col in shuffled.columns:
        shuffled[col] = np.random.permutation(shuffled[col].values)

    # Concatenate intruders
    intruders = pd.concat([noisy, outliers, shuffled], ignore_index=True)

    # Add labels
    legit = numeric_df.copy()
    legit["Label"] = 1
    intruders["Label"] = 0

    # Combine legit + intruders
    combined = pd.concat([legit, intruders], ignore_index=True)

    # If Timestamp exists in original, keep it only for legit users
    if "Timestamp" in df.columns:
        legit_ts = df["Timestamp"]
        legit["Timestamp"] = legit_ts.values
        intruders["Timestamp"] = pd.date_range(start="2025-08-01", periods=len(intruders), freq="S")
        combined = pd.concat([legit, intruders], ignore_index=True)

    # Save
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Intruder data generated and saved to: {OUTPUT_CSV}")
    print(f"   Legit rows: {len(legit)}, Intruder rows: {len(intruders)}")

if __name__ == "__main__":
    generate_intruders()