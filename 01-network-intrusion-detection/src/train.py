"""Train and evaluate intrusion detection models on NSL-KDD.

Two tasks:
  1. Binary  - is this connection an attack or normal traffic?
  2. Family  - which attack family (DoS, Probe, R2L, U2R) or Normal?

Usage:  python src/train.py
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import data

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
SEED = 42


def preprocessor(scale: bool) -> ColumnTransformer:
    numeric = [c for c in data.COLUMNS[:-2] if c not in data.CATEGORICAL]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), data.CATEGORICAL),
        ("num", StandardScaler() if scale else "passthrough", numeric),
    ])


def binary_models() -> dict:
    return {
        "logistic_regression": Pipeline([
            ("prep", preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("prep", preprocessor(scale=False)),
            ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                           class_weight="balanced", random_state=SEED)),
        ]),
    }


def evaluate_binary(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
    }


def plot_confusion(y_true, y_pred, labels, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, ax=ax, cmap="Blues",
        normalize="true", values_format=".2f", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importance(pipeline, path, top=15):
    names = pipeline.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    imp = pipeline.named_steps["clf"].feature_importances_
    order = np.argsort(imp)[-top:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([names[i] for i in order], imp[order], color="#3b6ea5")
    ax.set_title("Top features (random forest, binary task)")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def known_vs_novel(model, train, test, X_test) -> dict:
    """Detection rate on attack types seen in training vs. types that only appear in test."""
    seen = set(train["label"].unique())
    attacks = test["is_attack"] == 1
    pred = model.predict(X_test)
    known = attacks & test["label"].isin(seen)
    novel = attacks & ~test["label"].isin(seen)
    return {
        "known_attack_rows": int(known.sum()),
        "known_attack_recall": round(float(pred[known.values].mean()), 4),
        "novel_attack_rows": int(novel.sum()),
        "novel_attack_recall": round(float(pred[novel.values].mean()), 4),
        "novel_attack_types": sorted(test.loc[novel, "label"].unique().tolist()),
    }


def threshold_sweep(model, X, y) -> list:
    """Recall / false-alarm rate across decision thresholds."""
    proba = model.predict_proba(X)[:, 1]
    y = np.asarray(y)
    rows = []
    for t in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
        pred = proba >= t
        rows.append({
            "threshold": t,
            "recall": round(float(pred[y == 1].mean()), 4),
            "false_alarm_rate": round(float(pred[y == 0].mean()), 4),
            "precision": round(float(y[pred].mean()), 4) if pred.any() else 0.0,
        })
    return rows


def plot_threshold(rows, path):
    t = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(t, [r["recall"] for r in rows], "o-", label="attack detection rate (recall)", color="#3b6ea5")
    ax.plot(t, [r["false_alarm_rate"] for r in rows], "s-", label="false alarm rate", color="#c0504d")
    ax.invert_xaxis()
    ax.set_xlabel("decision threshold (lower = more sensitive)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Sensitivity trade-off (random forest, test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    train, test = data.load("train"), data.load("test")
    X_train, X_test = data.features(train), data.features(test)
    print(f"train: {len(train):,} rows | test: {len(test):,} rows")

    metrics = {"binary": {}, "family": {}}

    # 1. Binary: attack vs normal
    for name, model in binary_models().items():
        model.fit(X_train, train["is_attack"])
        metrics["binary"][name] = evaluate_binary(model, X_test, test["is_attack"])
        print(f"[binary] {name}: {metrics['binary'][name]}")
        if name == "random_forest":
            joblib.dump(model, MODELS / "binary_rf.joblib")
            pred = model.predict(X_test)
            plot_confusion(test["is_attack"].map({0: "Normal", 1: "Attack"}),
                           np.where(pred == 1, "Attack", "Normal"),
                           ["Normal", "Attack"], "Binary task (test set)",
                           RESULTS / "confusion_binary.png")
            plot_importance(model, RESULTS / "feature_importance.png")
            metrics["generalization"] = known_vs_novel(model, train, test, X_test)
            print(f"[novel attacks] {metrics['generalization']}")
            metrics["threshold_sweep"] = threshold_sweep(model, X_test, test["is_attack"])
            plot_threshold(metrics["threshold_sweep"], RESULTS / "threshold_tradeoff.png")

    # 2. Attack family classification
    family_model = Pipeline([
        ("prep", preprocessor(scale=False)),
        ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                       class_weight="balanced_subsample", random_state=SEED)),
    ])
    family_model.fit(X_train, train["family"])
    pred = family_model.predict(X_test)
    labels = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    report = classification_report(test["family"], pred, labels=labels,
                                   output_dict=True, zero_division=0)
    metrics["family"] = {
        "accuracy": round(report["accuracy"], 4) if "accuracy" in report else round(accuracy_score(test["family"], pred), 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "per_class_recall": {k: round(report[k]["recall"], 4) for k in labels},
    }
    print(f"[family] {metrics['family']}")
    joblib.dump(family_model, MODELS / "family_rf.joblib")
    plot_confusion(test["family"], pred, labels, "Attack family (test set)",
                   RESULTS / "confusion_family.png")

    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved results to {RESULTS}")


if __name__ == "__main__":
    main()
