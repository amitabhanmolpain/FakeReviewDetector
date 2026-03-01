import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- Setup FastAPI App ---
app = FastAPI()

# Enable CORS for Chrome Extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define Data Structure ---
class ReviewRequest(BaseModel):
    review: str

class PredictionResponse(BaseModel):
    prediction: str

# --- Load ML Model & Vectorizer ---
# Adjust paths if your root folder differs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fake_review_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Logic loaded: Model and Vectorizer are ready.")
except Exception as e:
    print(f"❌ Error loading models: {str(e)}")
    model = None
    vectorizer = None

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "TrustScan API is up and running!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Machine Learning model not loaded on server.")
    
    try:
        # Preprocessing & Prediction
        # 1. Vectorize text
        vectorized_text = vectorizer.transform([request.review])
        
        # 2. Predict
        prediction = model.predict(vectorized_text)[0]
        
        # 3. Map result to requirement (assuming model returns 0/1 or OR/CG)
        # Note: Adjust logic if your model uses different label mappings
        result = str(prediction)
        
        # Mapping to specific requirement: CG=Fake, OR=Genuine
        # If your model returns 1 for Fake and 0 for Genuine:
        if result == "1" or result.upper() == "CG":
            final_label = "CG"
        else:
            final_label = "OR"
            
        return PredictionResponse(prediction=final_label)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

# --- Run App ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
