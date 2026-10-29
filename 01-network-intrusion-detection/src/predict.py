"""Score network connection records with the trained models.

Usage:
  python src/predict.py path/to/connections.csv [--threshold 0.2]

The CSV must have the 41 NSL-KDD feature columns (see data.COLUMNS).
With no path, it scores 10 random rows from the test set as a demo.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

import data

MODELS = Path(__file__).resolve().parent.parent / "models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="attack probability needed to raise an alert")
    args = parser.parse_args()

    binary = joblib.load(MODELS / "binary_rf.joblib")
    family = joblib.load(MODELS / "family_rf.joblib")

    if args.csv:
        rows = pd.read_csv(args.csv)
        truth = None
    else:
        sample = data.load("test").sample(10, random_state=7)
        rows, truth = data.features(sample), sample["label"].tolist()

    proba = binary.predict_proba(rows)[:, 1]
    fam = family.predict(rows)
    for i, (p, f) in enumerate(zip(proba, fam)):
        verdict = "ALERT " if p >= args.threshold else "ok    "
        extra = f"  (true label: {truth[i]})" if truth else ""
        print(f"{verdict} p(attack)={p:.2f}  family={f:<6}{extra}")


if __name__ == "__main__":
    main()
