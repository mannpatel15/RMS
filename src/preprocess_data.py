# Implementation/src/preprocess_data.py
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv("base_dir")
INPUT = os.path.join(base_dir, "data", "cd_vector.csv")
OUTPUT = os.path.join(base_dir, "data", "cd_vector_clean.csv")

def preprocess():
    df = pd.read_csv(INPUT)

    # Drop fully empty columns
    df = df.dropna(axis=1, how="all")

    # Keep numeric feature columns; keep Label/Timestamp only for reference
    num = df.select_dtypes(include="number")
    out = num.copy()

    # Put Label back if present (for evaluation only)
    if "Label" in df.columns:
        out["Label"] = df["Label"]

    # Keep Timestamp only for traceability
    if "Timestamp" in df.columns:
        out["Timestamp"] = df["Timestamp"]

    # Fill numeric NaNs with column means
    num_cols = num.columns
    out[num_cols] = out[num_cols].fillna(out[num_cols].mean())

    out.to_csv(OUTPUT, index=False)
    print(f"✅ Clean data saved: {OUTPUT}")

if __name__ == "__main__":
    preprocess()