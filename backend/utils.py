"""
backend/utils.py
Helpers: preprocessing, heuristics, model load/save, data loading.
"""

import os
import re
import string
import json
import joblib
import numpy as np
import pandas as pd

# ─── Thresholds & Configuration ────────────────────────────────────────────
FAKE_THRESHOLD = 0.85
REAL_THRESHOLD = 0.35
ML_WEIGHT = 0.75

HEURISTIC_WEIGHTS = {
    "promo_phrase": 0.40,
    "repeated_word_ratio": 0.20,
    "short_review": 0.25,
    "exclamation_ratio": 0.15,
    "uppercase_ratio": 0.10,
}
# Weights sum to 1.10 – normalise once at import time
_HW_SUM = sum(HEURISTIC_WEIGHTS.values())
HEURISTIC_WEIGHTS_NORM = {k: v / _HW_SUM for k, v in HEURISTIC_WEIGHTS.items()}

# Promotional phrases that are strong fake‑review signals
PROMO_PHRASES = [
    "best product ever",
    "highly recommend",
    "must buy",
    "five stars",
    "5 stars",
    "amazing product",
    "love it so much",
    "you won't regret",
    "buy this now",
    "changed my life",
    "game changer",
    "absolutely perfect",
    "works like a charm",
    "don't hesitate",
    "100% satisfied",
    "totally worth it",
    "best purchase ever",
    "incredible quality",
    "free product",
    "received this product for free",
    "in exchange for my honest review",
    "discount code",
    "use my code",
    "click the link",
]

# ─── NLTK bootstrap ───────────────────────────────────────────────────────
_STOPWORDS = None


def _ensure_nltk():
    """Download NLTK data once, silently."""
    global _STOPWORDS
    if _STOPWORDS is not None:
        return
    try:
        import nltk
        from nltk.corpus import stopwords as _sw

        try:
            _STOPWORDS = set(_sw.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            _STOPWORDS = set(_sw.words("english"))
    except Exception:
        _STOPWORDS = set()


# ─── Text preprocessing ───────────────────────────────────────────────────
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def preprocess_text(text: str) -> str:
    """Clean a single review string for model input."""
    if not isinstance(text, str):
        return ""
    _ensure_nltk()
    text = text.lower()
    text = _URL_RE.sub(" ", text)  # remove URLs
    text = text.translate(_PUNCT_TABLE)  # remove punctuation
    tokens = text.split()
    tokens = [t for t in tokens if not t.isdigit()]  # drop pure digits
    tokens = [t for t in tokens if t not in _STOPWORDS]  # remove stopwords
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


# ─── Column detection ─────────────────────────────────────────────────────
TEXT_CANDIDATES = ["text_", "review", "review_text", "reviewText", "text"]
LABEL_CANDIDATES = ["label", "target"]

# Label value mapping
FAKE_VALUES = {"cg", "fake", "fraud", "1", "computer generated"}
REAL_VALUES = {"or", "real", "genuine", "0", "original"}


def detect_text_label_columns(df: pd.DataFrame):
    """Return (text_col, label_col) or raise with a clear message."""
    cols_lower = {c.lower(): c for c in df.columns}

    text_col = None
    for cand in TEXT_CANDIDATES:
        if cand.lower() in cols_lower:
            text_col = cols_lower[cand.lower()]
            break
    if text_col is None:
        raise ValueError(
            f"No text column found. Looked for {TEXT_CANDIDATES} in {list(df.columns)}"
        )

    label_col = None
    for cand in LABEL_CANDIDATES:
        if cand.lower() in cols_lower:
            label_col = cols_lower[cand.lower()]
            break
    if label_col is None:
        raise ValueError(
            f"No label column found. Looked for {LABEL_CANDIDATES} in {list(df.columns)}"
        )

    return text_col, label_col


def map_labels(series: pd.Series) -> pd.Series:
    """Map raw label values to 'fake' / 'real'. Raise on unknown."""
    mapped = series.astype(str).str.strip().str.lower()
    unique = set(mapped.unique())
    unknown = unique - FAKE_VALUES - REAL_VALUES
    if unknown:
        raise ValueError(
            f"Unknown label values: {unknown}. "
            f"Known fake={FAKE_VALUES}, real={REAL_VALUES}"
        )
    mapped = mapped.map(lambda v: "fake" if v in FAKE_VALUES else "real")
    return mapped


# ─── Data loading ──────────────────────────────────────────────────────────
def load_data(path: str):
    """Load CSV and return (df, text_col, label_col)."""
    df = pd.read_csv(path)
    text_col, label_col = detect_text_label_columns(df)
    return df, text_col, label_col


# ─── Heuristics ───────────────────────────────────────────────────────────
def heuristics_score(text: str) -> dict:
    """
    Compute lightweight heuristic signals for a review.
    Returns dict with individual signals and a combined heuristic_score (0-1).
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            "promo_phrase": False,
            "repeated_word_ratio": 0.0,
            "short_review": True,
            "exclamation_ratio": 0.0,
            "uppercase_ratio": 0.0,
            "heuristic_score": 0.5,
        }

    text_lower = text.lower()
    tokens = text.split()
    total_tokens = max(len(tokens), 1)

    # 1. Promotional phrase detection
    promo = any(p in text_lower for p in PROMO_PHRASES)

    # 2. Repeated word ratio
    unique_tokens = set(t.lower() for t in tokens)
    repeated_ratio = 1.0 - (len(unique_tokens) / total_tokens) if total_tokens > 1 else 0.0

    # 3. Short review
    short = total_tokens < 6

    # 4. Exclamation ratio
    excl_count = text.count("!")
    excl_ratio = min(excl_count / max(len(text), 1), 1.0)

    # 5. Uppercase ratio (tokens that are fully uppercase and len > 1)
    upper_tokens = sum(1 for t in tokens if t.isupper() and len(t) > 1)
    upper_ratio = upper_tokens / total_tokens

    # Combine into single 0-1 score
    w = HEURISTIC_WEIGHTS_NORM
    score = (
        w["promo_phrase"] * float(promo)
        + w["repeated_word_ratio"] * repeated_ratio
        + w["short_review"] * float(short)
        + w["exclamation_ratio"] * min(excl_ratio * 10, 1.0)  # scale up
        + w["uppercase_ratio"] * min(upper_ratio * 5, 1.0)  # scale up
    )
    score = float(np.clip(score, 0.0, 1.0))

    return {
        "promo_phrase": promo,
        "repeated_word_ratio": round(repeated_ratio, 4),
        "short_review": short,
        "exclamation_ratio": round(excl_ratio, 4),
        "uppercase_ratio": round(upper_ratio, 4),
        "heuristic_score": round(score, 4),
    }


def combine_scores(
    ml_prob_fake: float,
    heuristic_score_val: float,
    ml_weight: float = ML_WEIGHT,
) -> float:
    """Weighted combination of ML probability and heuristic score."""
    combined = ml_weight * ml_prob_fake + (1 - ml_weight) * heuristic_score_val
    return float(np.clip(combined, 0.0, 1.0))


# ─── Model helpers ─────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def pipeline_path():
    return os.path.join(MODELS_DIR, "fake_review_pipeline.pkl")


def metadata_path():
    return os.path.join(MODELS_DIR, "metadata.json")


def metrics_path():
    return os.path.join(MODELS_DIR, "metrics.json")


def load_pipeline():
    """Load the sklearn Pipeline from disk."""
    p = pipeline_path()
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"Pipeline not found at {p}. Run: python backend/train.py"
        )
    return joblib.load(p)


def load_legacy_model_vectorizer():
    """Fallback: load old separate model + vectorizer files."""
    model_p = os.path.join(MODELS_DIR, "fake_review_model.pkl")
    vec_p = os.path.join(MODELS_DIR, "vectorizer.pkl")
    if not os.path.isfile(model_p) or not os.path.isfile(vec_p):
        raise FileNotFoundError(
            "Neither pipeline nor legacy model/vectorizer found in models/."
        )
    model = joblib.load(model_p)
    vectorizer = joblib.load(vec_p)
    return model, vectorizer


def save_metadata(meta: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(metadata_path(), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def save_metrics(metrics: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(metrics_path(), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
