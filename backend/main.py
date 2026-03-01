"""
backend/main.py
FastAPI app serving fake‑review predictions.

Run:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils import (
    FAKE_THRESHOLD,
    ML_WEIGHT,
    REAL_THRESHOLD,
    combine_scores,
    heuristics_score,
    load_legacy_model_vectorizer,
    load_pipeline,
    preprocess_text,
)

# ─── App creation ──────────────────────────────────────────────────────────
app = FastAPI(title="TrustScan – Fake Review Detector API", version="1.0")

# CORS – allow localhost origins for extension dev + any origin for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global model holder ──────────────────────────────────────────────────
_pipeline = None
_pipeline_classes = None
_using_legacy = False


def _load_model():
    """Try loading pipeline; fallback to legacy model + vectorizer."""
    global _pipeline, _pipeline_classes, _using_legacy

    # Prefer the new pipeline
    try:
        _pipeline = load_pipeline()
        _pipeline_classes = _pipeline.classes_
        _using_legacy = False
        print("✅  Pipeline loaded (fake_review_pipeline.pkl)")
        return
    except FileNotFoundError:
        pass

    # Fallback to legacy separate files
    try:
        model, vectorizer = load_legacy_model_vectorizer()
        # Wrap into a pseudo‑pipeline (duck‑typing)
        from sklearn.pipeline import make_pipeline

        _pipeline = make_pipeline(vectorizer, model)
        _pipeline_classes = (
            model.classes_ if hasattr(model, "classes_") else np.array(["fake", "real"])
        )
        _using_legacy = True
        print("✅  Legacy model + vectorizer loaded as pipeline fallback")
    except FileNotFoundError as exc:
        print(f"❌  {exc}")
        print("   Train a model first:  python backend/train.py")


@app.on_event("startup")
def startup_event():
    _load_model()


# ─── Request / Response schemas ───────────────────────────────────────────
class ReviewRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must be a non‑empty string")
        return v


class PredictionResponse(BaseModel):
    label: str
    fake_prob: float
    combined_fake_score: float
    confidence: float
    heuristics: Dict[str, Any]
    meta: Dict[str, Any]


# ─── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "TrustScan API is up and running!"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):
    if _pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Run 'python backend/train.py' first.",
        )

    raw_text = request.text
    cleaned = preprocess_text(raw_text)

    # ── ML probability ─────────────────────────────────────────────────
    try:
        probs = _pipeline.predict_proba([cleaned])[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")

    # Determine index for the 'fake' class
    classes_list = list(_pipeline_classes)
    if "fake" in classes_list:
        fake_idx = classes_list.index("fake")
    elif "CG" in classes_list:
        fake_idx = classes_list.index("CG")
    elif 1 in classes_list:
        fake_idx = classes_list.index(1)
    else:
        fake_idx = 0  # best guess

    ml_prob_fake = float(probs[fake_idx])

    # ── Heuristics ─────────────────────────────────────────────────────
    h = heuristics_score(raw_text)
    heuristic_val = h["heuristic_score"]

    # ── Combined score ─────────────────────────────────────────────────
    combined = combine_scores(ml_prob_fake, heuristic_val, ml_weight=ML_WEIGHT)

    # ── Label assignment ───────────────────────────────────────────────
    if combined >= FAKE_THRESHOLD:
        label = "Fake"
    elif combined <= REAL_THRESHOLD:
        label = "Real"
    else:
        label = "Uncertain"

    confidence = combined if label == "Fake" else (1 - combined) if label == "Real" else max(combined, 1 - combined)

    return PredictionResponse(
        label=label,
        fake_prob=round(ml_prob_fake, 4),
        combined_fake_score=round(combined, 4),
        confidence=round(confidence, 4),
        heuristics=h,
        meta={
            "model_version": "v1",
            "used_vectorizer": "tfidf",
            "legacy_fallback": _using_legacy,
            "note": f"thresholds: fake>={FAKE_THRESHOLD}, real<={REAL_THRESHOLD}, ml_weight={ML_WEIGHT}",
        },
    )


# ─── Run directly ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
