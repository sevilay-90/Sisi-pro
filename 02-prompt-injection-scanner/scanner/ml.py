"""Machine learning layer: a TF-IDF + logistic regression text classifier.

Rules catch known phrasings precisely; the classifier generalizes to
rephrased attacks the rules have never seen.
"""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL_PATH = ROOT / "models" / "classifier.joblib"


def load_dataset():
    texts, labels = [], []
    for name, label in (("injections.txt", 1), ("benign.txt", 0)):
        for line in (DATA / name).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                texts.append(line)
                labels.append(label)
    return texts, labels


def build_model() -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")),
    ])


def train(texts=None, labels=None, save=True) -> Pipeline:
    if texts is None:
        texts, labels = load_dataset()
    model = build_model().fit(texts, labels)
    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
    return model


_cached = None


def get_model() -> Pipeline:
    """Load the saved model, training it first if it does not exist yet."""
    global _cached
    if _cached is None:
        _cached = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else train()
    return _cached
