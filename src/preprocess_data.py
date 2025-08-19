# Implementation/src/preprocess_data.py
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Implementation/
DATA_DIR = os.path.join(BASE_DIR, "data")

# Input & output file paths
INPUT_CSV = os.path.join(DATA_DIR, "cd_vector.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "cd_vectors_clean.csv")

def preprocess_data(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    # Load dataset
    df = pd.read_csv(input_csv)

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # Fill missing numeric values with column mean
    df = df.fillna(df.mean(numeric_only=True))

    # Select features (ignore Timestamp, keep Label as target)
    if "Label" not in df.columns or "Timestamp" not in df.columns:
        raise ValueError("❌ CSV must contain 'Label' and 'Timestamp' columns.")

    features = df.drop(columns=["Label", "Timestamp"])
    labels = df["Label"]

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Create processed DataFrame
    processed_df = pd.DataFrame(scaled_features, columns=features.columns)
    processed_df["Label"] = labels.values

    # Save cleaned data
    processed_df.to_csv(output_csv, index=False)
    print(f"✅ Preprocessed data saved to {output_csv}")

    return processed_df

if __name__ == "__main__":
    preprocess_data()
