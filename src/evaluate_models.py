# Implementation/src/evaluate_models.py
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv("base_dir")
PRED = os.path.join(base_dir, "data", "ensemble_predictions.csv")

def evaluate():
    df = pd.read_csv(PRED)
    if "Label" not in df.columns:
        print("⚠️ No Label column; cannot compute supervised metrics.")
        print(f"Flag rate: {df['EnsemblePrediction'].mean():.4f}")
        return

    y_true = df["Label"]
    y_pred = df["EnsemblePrediction"]

    classes = sorted(y_true.dropna().unique().tolist())
    if len(classes) < 2:
        print("⚠️ Only one class present in labels. Supervised metrics are not meaningful.")
        print(f"Owner count: {(y_true==1).sum()} | Intruder count: {(y_true==0).sum()}")
        print(f"Flag rate: {y_pred.mean():.4f}")
        # still show a 2x2 confusion matrix layout
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        print("\nConfusion Matrix (rows=true, cols=pred, labels=[0,1])")
        print(cm)
        return

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    print("📊 Evaluation Metrics")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("\n🔎 Confusion Matrix (rows=true, cols=pred)")
    print(cm)
    print("\n📄 Classification Report")
    print(classification_report(y_true, y_pred, target_names=["Intruder(0)","Owner(1)"], zero_division=0))

if __name__ == "__main__":
    evaluate()