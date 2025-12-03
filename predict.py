import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import xgboost as xgb
import numpy as np
from typing import Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Encoding dictionaries
NEIGHBOURHOOD_ENCODING = {
    "manhattan": 1,
    "brooklyn": 0,
    "queens": 2,
    "bronx": 4,
    "staten island": 3,
    "staten_island": 3,
}

ROOM_TYPE_ENCODING = {
    "entire home/apt": 1,
    "entire_home/apt": 1,
    "private room": 0,
    "private_room": 0,
    "shared room": 2,
    "shared_room": 2,
}

class ListingRequest(BaseModel):
    neighbourhood_group: int
    room_type: int
    minimum_nights: int
    calculated_host_listings_count: int
    availability_365: int

    @field_validator("neighbourhood_group", mode="before")
    def encode_neighbourhood(cls, v):
        if isinstance(v, int):
            return v
        key = str(v).strip().lower().replace("_", " ")
        if key not in NEIGHBOURHOOD_ENCODING:
            raise ValueError(f"Unknown neighbourhood_group: {v}. Valid options: {list(NEIGHBOURHOOD_ENCODING.keys())}")
        return NEIGHBOURHOOD_ENCODING[key]

    @field_validator("room_type", mode="before")
    def encode_room_type(cls, v):
        if isinstance(v, int):
            return v
        key = str(v).strip().lower().replace("_", " ")
        if key not in ROOM_TYPE_ENCODING:
            raise ValueError(f"Unknown room_type: {v}. Valid options: {list(ROOM_TYPE_ENCODING.keys())}")
        return ROOM_TYPE_ENCODING[key]

class PredictResponse(BaseModel):
    price_prediction: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

# Initialize FastAPI app
app = FastAPI(
    title="NYC Airbnb Price Prediction API",
    description="Predict Airbnb listing prices in New York City",
    version="1.0.0"
)

# Add CORS middleware (allows requests from web browsers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
try:
    with open('model.bin', 'rb') as f_in:
        dv, pipeline = pickle.load(f_in)
    logger.info("Model loaded successfully")
    MODEL_LOADED = True
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    MODEL_LOADED = False
    dv, pipeline = None, None

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "NYC Airbnb Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if MODEL_LOADED else "unhealthy",
        model_loaded=MODEL_LOADED,
        version="1.0.0"
    )

@app.post('/predict', response_model=PredictResponse)
def predict(request: ListingRequest):
    """
    Predict Airbnb listing price
    
    Example request:
    {
        "neighbourhood_group": "manhattan",
        "room_type": "entire home/apt",
        "minimum_nights": 3,
        "calculated_host_listings_count": 5,
        "availability_365": 200
    }
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Transform input
        X = dv.transform([request.model_dump()])
        
        # Get feature names
        features = list(dv.get_feature_names_out())
        
        # Create DMatrix for XGBoost
        dX = xgb.DMatrix(X, feature_names=features)
        
        # Predict (model outputs log-transformed price)
        y_pred = pipeline.predict(dX)
        
        # Reverse log transformation
        price = float(y_pred[0])
        price = round(np.exp(price), 2)
        
        logger.info(f"Prediction made: ${price}")
        
        return PredictResponse(price_prediction=price)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/encodings")
def get_encodings() -> Dict:
    """Get valid encoding options for categorical features"""
    return {
        "neighbourhood_group": list(NEIGHBOURHOOD_ENCODING.keys()),
        "room_type": list(ROOM_TYPE_ENCODING.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=9696)