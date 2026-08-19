from pathlib import Path
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from model_utils import save_model

def main(data_path: str):
    df = pd.read_csv(data_path)

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Simple, solid baseline
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy: {acc:.3f}")
    print(f"Macro-F1: {f1:.3f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    save_model(pipe)
    print("\nSaved model to artifacts/model.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path("data")/"Crop_recommendation.csv"))
    args = parser.parse_args()
    main(args.data)
