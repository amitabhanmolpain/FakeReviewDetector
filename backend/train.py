"""
backend/train.py
Training script – produces models/fake_review_pipeline.pkl

Usage:
    python backend/train.py            # train only if pipeline does not exist
    python backend/train.py --force    # retrain and overwrite
    python backend/train.py --retrain  # same as --force
"""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Ensure backend package is importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils import (
    MODELS_DIR,
    load_data,
    map_labels,
    pipeline_path,
    preprocess_text,
    save_metadata,
)

# ─── Paths ─────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "fakereviews.csv",
)
LEGACY_VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
LEGACY_MODEL_PATH = os.path.join(MODELS_DIR, "fake_review_model.pkl")


def parse_args():
    parser = argparse.ArgumentParser(description="Train fake‑review detection pipeline")
    parser.add_argument(
        "--force",
        "--retrain",
        action="store_true",
        dest="force",
        help="Retrain even if a pipeline already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Guard: skip if pipeline exists and --force not given ───────────────
    pp = pipeline_path()
    if os.path.isfile(pp) and not args.force:
        print(f"[INFO] Pipeline already exists at {pp}")
        print("       Run with --force or --retrain to overwrite.")
        return

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"[INFO] Loading data from {DATA_PATH} ...")
    df, text_col, label_col = load_data(DATA_PATH)
    print(f"       Shape: {df.shape}  |  text_col={text_col}  |  label_col={label_col}")

    # ── Map labels ─────────────────────────────────────────────────────────
    df["_label"] = map_labels(df[label_col])
    class_counts = df["_label"].value_counts().to_dict()
    print(f"[INFO] Class distribution: {class_counts}")

    # ── Preprocess ─────────────────────────────────────────────────────────
    print("[INFO] Preprocessing text ...")
    df["_clean"] = df[text_col].apply(preprocess_text)

    X = df["_clean"].values
    y = df["_label"].values

    # ── Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"[INFO] Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # ── Build pipeline ─────────────────────────────────────────────────────
    # Check for a compatible legacy vectorizer that can seed params
    tfidf_params = dict(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
    )

    if os.path.isfile(LEGACY_VECTORIZER_PATH) and not args.force:
        try:
            legacy_vec = joblib.load(LEGACY_VECTORIZER_PATH)
            print(f"[INFO] Found legacy vectorizer at {LEGACY_VECTORIZER_PATH} (noted, training fresh pipeline)")
        except Exception:
            pass

    pipeline = make_pipeline(
        TfidfVectorizer(**tfidf_params),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("[INFO] Training pipeline ...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[INFO] Training completed in {elapsed:.2f}s")

    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    print(f"[INFO] Train accuracy: {train_acc:.4f}  |  Test accuracy: {test_acc:.4f}")

    # ── Top‑20 features for the 'fake' class ──────────────────────────────
    vectorizer = pipeline.named_steps["tfidfvectorizer"]
    clf = pipeline.named_steps["logisticregression"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = clf.classes_
    fake_idx = list(classes).index("fake")
    coeffs = clf.coef_[fake_idx]
    top_k = 20
    top_indices = np.argsort(coeffs)[-top_k:][::-1]

    print(f"\n[INFO] Top {top_k} features for 'fake' class:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i:>2}. {feature_names[idx]:<30s}  coeff={coeffs[idx]:.4f}")

    # ── Save pipeline ──────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, pp)
    print(f"\n[INFO] Pipeline saved to {pp}")

    # ── Save test indices so evaluate.py can reproduce the same split ──────
    test_indices_path = os.path.join(MODELS_DIR, "test_indices.json")
    # We save the original df indices corresponding to the test set rows
    # Since we used stratified split on arrays, we recover indices via positions
    all_indices = np.arange(len(df))
    _, test_idx_arr = train_test_split(
        all_indices, test_size=0.2, stratify=y, random_state=42
    )
    with open(test_indices_path, "w", encoding="utf-8") as f:
        json.dump(test_idx_arr.tolist(), f)
    print(f"[INFO] Test indices saved to {test_indices_path}")

    # ── Save metadata ──────────────────────────────────────────────────────
    meta = {
        "model_type": "LogisticRegression",
        "vectorizer_type": "TfidfVectorizer",
        "vectorizer_params": tfidf_params,
        "label_mapping": {"CG": "fake", "OR": "real"},
        "classes": classes.tolist(),
        "training_class_counts": class_counts,
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "training_time_sec": round(elapsed, 2),
        "random_state": 42,
        "test_size": 0.2,
    }
    save_metadata(meta)
    print("[INFO] Metadata saved to models/metadata.json")
    print("\n✅  Training complete.")


if __name__ == "__main__":
    main()
