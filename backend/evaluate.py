"""
backend/evaluate.py
Evaluation script – loads the trained pipeline and evaluates on the held‑out test set.

Usage:
    python backend/evaluate.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils import (
    MODELS_DIR,
    load_data,
    load_pipeline,
    map_labels,
    preprocess_text,
    save_metrics,
)

# ─── Configurable thresholds ──────────────────────────────────────────────
FAKE_THRESHOLD = 0.85
REAL_THRESHOLD = 0.35

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "fakereviews.csv",
)
TEST_INDICES_PATH = os.path.join(MODELS_DIR, "test_indices.json")


def three_class_label(prob_fake: float) -> str:
    """Apply thresholds to produce Fake / Real / Uncertain."""
    if prob_fake >= FAKE_THRESHOLD:
        return "Fake"
    elif prob_fake <= REAL_THRESHOLD:
        return "Real"
    return "Uncertain"


def get_confident_false_positives(
    df: pd.DataFrame,
    probs_fake: np.ndarray,
    true_labels: np.ndarray,
    n: int = 20,
):
    """Print reviews predicted fake with high confidence but actually real."""
    fp_mask = (probs_fake >= FAKE_THRESHOLD) & (true_labels == "real")
    fp_indices = np.where(fp_mask)[0]
    # sort by descending fake prob
    fp_sorted = fp_indices[np.argsort(probs_fake[fp_indices])[::-1]][:n]

    print(f"\n{'='*80}")
    print(f"Top {len(fp_sorted)} CONFIDENT FALSE POSITIVES (predicted Fake, actual Real)")
    print(f"{'='*80}")
    for rank, idx in enumerate(fp_sorted, 1):
        text_snippet = str(df.iloc[idx].get("text_", df.iloc[idx].values[0]))[:120]
        print(f"  {rank:>2}. prob={probs_fake[idx]:.4f}  |  {text_snippet}...")
    return fp_sorted


def main():
    # ── Load pipeline ──────────────────────────────────────────────────────
    print("[INFO] Loading pipeline ...")
    pipeline = load_pipeline()

    # ── Load data & reproduce test split ───────────────────────────────────
    print(f"[INFO] Loading data from {DATA_PATH} ...")
    df, text_col, label_col = load_data(DATA_PATH)
    df["_label"] = map_labels(df[label_col])

    if os.path.isfile(TEST_INDICES_PATH):
        with open(TEST_INDICES_PATH, "r", encoding="utf-8") as f:
            test_indices = json.load(f)
        df_test = df.iloc[test_indices].copy()
        print(f"[INFO] Loaded {len(test_indices)} test indices from {TEST_INDICES_PATH}")
    else:
        # Fallback: reproduce split
        from sklearn.model_selection import train_test_split

        _, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=0.2,
            stratify=df["_label"].values,
            random_state=42,
        )
        df_test = df.iloc[test_idx].copy()
        print(f"[INFO] Reproduced test split ({len(df_test)} samples)")

    # ── Preprocess & predict ───────────────────────────────────────────────
    print("[INFO] Preprocessing & predicting ...")
    X_test = df_test[text_col].apply(preprocess_text).values
    y_true = df_test["_label"].values

    classes = pipeline.classes_
    fake_idx = list(classes).index("fake")

    probs = pipeline.predict_proba(X_test)
    probs_fake = probs[:, fake_idx]

    # Binary predictions (for standard metrics)
    y_pred_binary = pipeline.predict(X_test)

    # Three‑class predictions
    y_pred_three = np.array([three_class_label(p) for p in probs_fake])

    # ── Classification report (binary) ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("BINARY CLASSIFICATION REPORT (fake vs real)")
    print("=" * 60)
    report_str = classification_report(y_true, y_pred_binary, digits=4)
    print(report_str)
    report_dict = classification_report(y_true, y_pred_binary, output_dict=True)

    # ── Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred_binary, labels=["fake", "real"])
    print("Confusion Matrix (rows=true, cols=pred) [fake, real]:")
    print(cm)

    # ── ROC AUC ────────────────────────────────────────────────────────────
    y_true_bin = (y_true == "fake").astype(int)
    auc = roc_auc_score(y_true_bin, probs_fake)
    print(f"\nROC AUC: {auc:.4f}")

    # ── Three‑class distribution ───────────────────────────────────────────
    unique, counts = np.unique(y_pred_three, return_counts=True)
    print(f"\nThree‑class distribution (thresholds: fake>={FAKE_THRESHOLD}, real<={REAL_THRESHOLD}):")
    for lbl, cnt in zip(unique, counts):
        print(f"  {lbl}: {cnt}")

    # ── Confident false positives ──────────────────────────────────────────
    get_confident_false_positives(df_test, probs_fake, y_true, n=20)

    # ── Save metrics JSON ──────────────────────────────────────────────────
    metrics = {
        "thresholds": {
            "fake_threshold": FAKE_THRESHOLD,
            "real_threshold": REAL_THRESHOLD,
        },
        "binary_classification_report": {
            "fake": {
                k: round(v, 4)
                for k, v in report_dict.get("fake", {}).items()
            },
            "real": {
                k: round(v, 4)
                for k, v in report_dict.get("real", {}).items()
            },
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["fake", "real"],
        "roc_auc": round(auc, 4),
        "three_class_counts": dict(zip(unique.tolist(), counts.tolist())),
    }
    save_metrics(metrics)
    print(f"\n[INFO] Metrics saved to models/metrics.json")
    print("\n✅  Evaluation complete.")


if __name__ == "__main__":
    main()
