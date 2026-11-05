"""Compare rules-only, ML-only and hybrid detection with 5-fold cross-validation.

The ML model is retrained on each training fold, so every example is scored
by a model that never saw it.

Usage:  python evaluate.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from scanner import core, ml

RESULTS = Path(__file__).resolve().parent / "results"


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "false_positive_rate": round(fp / (fp + tn), 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    texts, labels = ml.load_dataset()
    texts, labels = np.array(texts, dtype=object), np.array(labels)
    print(f"{len(texts)} examples ({labels.sum()} injections, {(1 - labels).sum()} benign)")

    preds = {"rules_only": np.zeros(len(texts), int),
             "ml_only": np.zeros(len(texts), int),
             "hybrid": np.zeros(len(texts), int)}
    mistakes = []

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in folds.split(texts, labels):
        ml._cached = ml.train(list(texts[train_idx]), list(labels[train_idx]), save=False)
        for i in test_idx:
            rules = core.scan(texts[i], use_ml=False)
            hybrid = core.scan(texts[i], use_ml=True)
            preds["rules_only"][i] = int(rules.score >= core.REVIEW_AT)
            preds["ml_only"][i] = int(hybrid.ml_score >= 0.5)
            preds["hybrid"][i] = int(hybrid.score >= core.REVIEW_AT)
            if preds["hybrid"][i] != labels[i]:
                kind = "missed injection" if labels[i] else "false alarm"
                mistakes.append({"type": kind, "score": hybrid.score, "text": texts[i]})

    report = {name: metrics(labels, p) for name, p in preds.items()}
    report["hybrid_mistakes"] = mistakes
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'method':<12}{'precision':>10}{'recall':>8}{'F1':>7}{'FPR':>7}")
    for name in preds:
        m = report[name]
        print(f"{name:<12}{m['precision']:>10}{m['recall']:>8}{m['f1']:>7}{m['false_positive_rate']:>7}")
    print("\nHybrid mistakes:")
    for m in mistakes:
        print(f"  [{m['type']}] {m['score']:.2f}  {m['text'][:90]}")

    ml._cached = None
    ml.train()  # final model on all data
    print("\nSaved final model to models/classifier.joblib")


if __name__ == "__main__":
    main()
