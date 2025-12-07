import pickle
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, Field
import xgboost as xgb
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import time
import os
from monitoring import model_monitor, get_monitor
from data_validator import data_validator, get_validator



# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

# Pydantic Models
class ListingRequest(BaseModel):
    """Request model for price prediction"""
    neighbourhood_group: int = Field(..., description="NYC borough (manhattan, brooklyn, queens, bronx, staten island)")
    room_type: int = Field(..., description="Type of room (entire home/apt, private room, shared room)")
    minimum_nights: int = Field(..., ge=0, le=365, description="Minimum nights required (0-365)")
    calculated_host_listings_count: int = Field(..., ge=0, description="Total listings by host")
    availability_365: int = Field(..., ge=0, le=365, description="Days available per year (0-365)")

    @field_validator("neighbourhood_group", mode="before")
    def encode_neighbourhood(cls, v):
        if isinstance(v, int):
            if v not in NEIGHBOURHOOD_ENCODING.values():
                raise ValueError(f"Invalid neighbourhood code: {v}")
            return v
        key = str(v).strip().lower().replace("_", " ")
        if key not in NEIGHBOURHOOD_ENCODING:
            valid_options = ', '.join(NEIGHBOURHOOD_ENCODING.keys())
            raise ValueError(f"Unknown neighbourhood_group: '{v}'. Valid options: {valid_options}")
        return NEIGHBOURHOOD_ENCODING[key]

    @field_validator("room_type", mode="before")
    def encode_room_type(cls, v):
        if isinstance(v, int):
            if v not in ROOM_TYPE_ENCODING.values():
                raise ValueError(f"Invalid room type code: {v}")
            return v
        key = str(v).strip().lower().replace("_", " ")
        if key not in ROOM_TYPE_ENCODING:
            valid_options = ', '.join(ROOM_TYPE_ENCODING.keys())
            raise ValueError(f"Unknown room_type: '{v}'. Valid options: {valid_options}")
        return ROOM_TYPE_ENCODING[key]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "neighbourhood_group": "manhattan",
                    "room_type": "entire home/apt",
                    "minimum_nights": 3,
                    "calculated_host_listings_count": 5,
                    "availability_365": 200
                }
            ]
        }
    }

class BatchListingRequest(BaseModel):
    """Request model for batch predictions"""
    listings: List[ListingRequest] = Field(..., max_length=100, description="List of listings (max 100)")

class PredictResponse(BaseModel):
    """Response model for single prediction"""
    price_prediction: float = Field(..., description="Predicted nightly price in USD")
    confidence: str = Field(default="medium", description="Prediction confidence level")
    input_features: Optional[Dict] = Field(default=None, description="Input features used")

class BatchPredictResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictResponse]
    total_listings: int
    processing_time_ms: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float
    total_predictions: int

class StatsResponse(BaseModel):
    """API statistics response"""
    total_predictions: int
    total_requests: int
    average_response_time_ms: float
    uptime_hours: float

# ==========================================
# NEW RESPONSE MODELS (Thêm sau PredictResponse)
# ==========================================

class PredictResponseWithValidation(BaseModel):
    """Enhanced prediction response with validation info"""
    price_prediction: float
    confidence: str
    input_features: Optional[Dict] = None
    data_quality: Optional[Dict] = None  # NEW: Quality report
    processing_time_ms: Optional[float] = None  # NEW: Processing time
    warnings: Optional[List[str]] = None  # NEW: Validation warnings

class MetricsResponse(BaseModel):
    """Monitoring metrics response"""
    total_predictions: int
    error_count: int
    error_rate: float
    confidence_distribution: Dict
    predictions_by_neighbourhood: Dict
    predictions_by_room_type: Dict
    price_statistics: Dict
    performance: Dict

class DashboardResponse(BaseModel):
    """Dashboard data response"""
    overview: Dict
    confidence_breakdown: Dict
    neighbourhood_breakdown: Dict
    room_type_breakdown: Dict
    price_range: Dict
    drift_detection: Dict
    recent_activity: List[Dict]


# Initialize FastAPI app
app = FastAPI(
    title="NYC Airbnb Price Prediction API",
    description="Machine Learning API for predicting Airbnb listing prices in New York City",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
START_TIME = time.time()
PREDICTION_COUNT = 0
REQUEST_COUNT = 0
TOTAL_RESPONSE_TIME = 0.0

# Load model at startup
try:
    with open('model.bin', 'rb') as f_in:
        dv, pipeline = pickle.load(f_in)
    logger.info("✅ Model loaded successfully")
    MODEL_LOADED = True
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    MODEL_LOADED = False
    dv, pipeline = None, None

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    global REQUEST_COUNT, TOTAL_RESPONSE_TIME
    
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # Convert to ms
    
    REQUEST_COUNT += 1
    TOTAL_RESPONSE_TIME += process_time
    
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    return response

# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": "Validation Error", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": "An unexpected error occurred"}
    )

# Endpoints
@app.get("/", tags=["Root"])
def root():
    """Root endpoint with API information"""
    return {
        "message": "NYC Airbnb Price Prediction API",
        "version": "1.0.0",
        "status": "healthy" if MODEL_LOADED else "degraded",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "batch_predict": "/predict/batch (POST)",
            "encodings": "/encodings (GET)",
            "stats": "/stats (GET)",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "documentation": "Visit /docs for interactive API documentation"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint for monitoring
    
    Returns the current health status of the API including:
    - Service status
    - Model loading status
    - API version
    - Uptime
    - Total predictions made
    """
    uptime = time.time() - START_TIME
    return HealthResponse(
        status="healthy" if MODEL_LOADED else "unhealthy",
        model_loaded=MODEL_LOADED,
        version="1.0.0",
        uptime_seconds=round(uptime, 2),
        total_predictions=PREDICTION_COUNT
    )

@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
def get_stats():
    """
    Get API usage statistics
    
    Returns:
    - Total predictions made
    - Total requests received
    - Average response time
    - Uptime in hours
    """
    uptime = (time.time() - START_TIME) / 3600  # Convert to hours
    avg_response_time = TOTAL_RESPONSE_TIME / REQUEST_COUNT if REQUEST_COUNT > 0 else 0
    
    return StatsResponse(
        total_predictions=PREDICTION_COUNT,
        total_requests=REQUEST_COUNT,
        average_response_time_ms=round(avg_response_time, 2),
        uptime_hours=round(uptime, 2)
    )


# ==========================================
# MONITORING ENDPOINTS
# Thêm các endpoints này vào cuối file predict.py
# (sau endpoint /price-range, trước if __name__)
# ==========================================

@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
def get_metrics():
    """
    Get comprehensive model and system metrics
    
    Returns detailed metrics including:
    - Total predictions made
    - Error rates
    - Confidence level distribution
    - Predictions breakdown by neighbourhood and room type
    - Price statistics (min, max, average)
    - Performance metrics (avg response time)
    
    **Example Response:**
    ```json
    {
        "total_predictions": 150,
        "error_count": 2,
        "error_rate": 0.013,
        "confidence_distribution": {
            "high": 80,
            "medium": 60,
            "low": 10
        },
        "price_statistics": {
            "min": 45.0,
            "max": 450.0,
            "average": 125.50
        }
    }
    ```
    """
    monitor = get_monitor()
    return monitor.get_metrics()


@app.get("/dashboard", response_model=DashboardResponse, tags=["Monitoring"])
def get_dashboard_data():
    """
    Get monitoring dashboard data
    
    Returns formatted data optimized for visualization dashboards:
    - Overview statistics
    - Confidence breakdown
    - Geographical distribution
    - Room type distribution
    - Price range analytics
    - Data drift detection results
    - Recent prediction activity
    
    **Use this endpoint for:**
    - Building monitoring dashboards
    - Generating reports
    - System health visualization
    """
    monitor = get_monitor()
    return monitor.get_dashboard_data()


@app.get("/drift", tags=["Monitoring"])
def check_data_drift():
    """
    Check for data drift in input features
    
    Compares recent input data distribution with baseline
    to detect potential model performance degradation.
    
    **Drift Detection Method:**
    - Compares first 100 predictions (baseline) vs last 100 (recent)
    - Calculates percentage change in feature averages
    - Flags drift if change exceeds 20%
    
    **Returns:**
    - Drift status for each feature (stable/drifted)
    - Baseline vs recent averages
    - Drift percentage
    
    **Example Response:**
    ```json
    {
        "drift_detection": {
            "minimum_nights": {
                "baseline_avg": 5.2,
                "recent_avg": 7.8,
                "drift_percentage": 50.0,
                "status": "drifted"
            },
            "availability_365": {
                "baseline_avg": 180,
                "recent_avg": 185,
                "drift_percentage": 2.78,
                "status": "stable"
            }
        },
        "timestamp": "2024-12-07T10:30:00",
        "status": "ok"
    }
    ```
    """
    monitor = get_monitor()
    drift_result = monitor.detect_data_drift()
    
    return {
        "drift_detection": drift_result,
        "timestamp": datetime.now().isoformat(),
        "status": "ok" if drift_result != {"status": "insufficient_data"} else "insufficient_data"
    }


@app.get("/recent-predictions", tags=["Monitoring"])
def get_recent_predictions(limit: int = 20):
    """
    Get recent prediction history
    
    **Query Parameters:**
    - limit: Number of predictions to return (default: 20, max: 100)
    
    **Returns:**
    List of recent predictions with:
    - Timestamp
    - Input features
    - Predicted price
    - Confidence level
    - Processing time
    
    **Example:**
    ```
    GET /recent-predictions?limit=10
    ```
    """
    monitor = get_monitor()
    
    # Cap limit at 100
    limit = min(limit, 100)
    
    return {
        "predictions": monitor.get_recent_predictions(limit),
        "total_in_history": len(monitor.prediction_history),
        "limit": limit
    }


@app.post("/export-metrics", tags=["Monitoring"])
def export_metrics():
    """
    Export current metrics to JSON file
    
    Saves a snapshot of current metrics to `metrics.json` file
    in the working directory.
    
    **Useful for:**
    - Periodic reporting
    - Offline analysis
    - Creating backups of metrics
    - Generating reports
    
    **Returns:**
    - Success/failure status
    - Export timestamp
    - File location
    """
    monitor = get_monitor()
    success = monitor.export_metrics("metrics.json")
    
    if success:
        return {
            "status": "success",
            "message": "Metrics exported successfully",
            "file": "metrics.json",
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(
            status_code=500, 
            detail="Failed to export metrics"
        )


@app.get("/data-quality", tags=["Monitoring"])
def check_data_quality(
    neighbourhood_group: str,
    room_type: str,
    minimum_nights: int,
    calculated_host_listings_count: int,
    availability_365: int
):
    """
    Check data quality without making prediction
    
    Validates input data and returns quality report without
    actually calling the ML model.
    
    **Query Parameters:**
    All the same parameters as /predict endpoint
    
    **Returns:**
    - Quality score (0-100)
    - Quality level (excellent/good/fair/poor)
    - Validation errors
    - Warnings
    - Outlier detection results
    - Recommendations
    
    **Example:**
    ```
    GET /data-quality?neighbourhood_group=manhattan&room_type=entire%20home/apt&minimum_nights=3&calculated_host_listings_count=5&availability_365=200
    ```
    """
    validator = get_validator()
    
    data = {
        "neighbourhood_group": neighbourhood_group,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365
    }
    
    quality_report = validator.get_data_quality_score(data)
    
    return quality_report


# ==========================================
# UPDATED PREDICT ENDPOINT
# Thay thế hàm predict() cũ bằng version này
# ==========================================

@app.post('/predict', response_model=PredictResponseWithValidation, tags=["Prediction"])
def predict(request: ListingRequest):
    """
    Predict Airbnb listing price with data validation and monitoring
    
    **Enhanced with:**
    - Data quality validation
    - Outlier detection
    - Performance monitoring
    - Detailed warnings
    
    **Example Request:**
    ```json
    {
        "neighbourhood_group": "manhattan",
        "room_type": "entire home/apt",
        "minimum_nights": 3,
        "calculated_host_listings_count": 5,
        "availability_365": 200
    }
    ```
    
    **Response:**
    ```json
    {
        "price_prediction": 157.42,
        "confidence": "medium",
        "input_features": {...},
        "data_quality": {
            "quality_score": 95,
            "quality_level": "excellent"
        },
        "processing_time_ms": 45.2,
        "warnings": []
    }
    ```
    """
    global PREDICTION_COUNT
    
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
    
    # Start timing
    start_time = time.time()
    
    try:
        # ==========================================
        # STEP 1: Data Validation
        # ==========================================
        validator = get_validator()
        input_data = request.model_dump()
        
        # Get quality report
        quality_report = validator.get_data_quality_score(input_data)
        
        # Check if data is valid
        if not quality_report["is_valid"]:
            # Data has errors - return error with details
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "Invalid input data",
                    "quality_report": quality_report
                }
            )
        
        # ==========================================
        # STEP 2: Model Prediction
        # ==========================================
        
        # Transform input
        X = dv.transform([input_data])
        
        # Get feature names
        features = list(dv.get_feature_names_out())
        
        # Create DMatrix for XGBoost
        dX = xgb.DMatrix(X, feature_names=features)
        
        # Predict (model outputs log-transformed price)
        y_pred = pipeline.predict(dX)
        
        # Reverse log transformation
        price = float(y_pred[0])
        price = round(np.exp(price), 2)
        
        # Determine confidence based on availability
        if request.availability_365 < 30:
            confidence = "low"
        elif request.availability_365 < 180:
            confidence = "medium"
        else:
            confidence = "high"
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # ==========================================
        # STEP 3: Log to Monitoring System
        # ==========================================
        monitor = get_monitor()
        monitor.log_prediction(
            input_features=input_data,
            prediction=price,
            confidence=confidence,
            processing_time_ms=processing_time_ms
        )
        
        # ==========================================
        # STEP 4: Increment Counter
        # ==========================================
        PREDICTION_COUNT += 1
        
        # ==========================================
        # STEP 5: Logging
        # ==========================================
        logger.info(
            f"Prediction made: ${price} (confidence: {confidence}, "
            f"quality: {quality_report['quality_level']}, "
            f"time: {processing_time_ms:.2f}ms)"
        )
        
        # ==========================================
        # STEP 6: Return Enhanced Response
        # ==========================================
        return PredictResponseWithValidation(
            price_prediction=price,
            confidence=confidence,
            input_features=input_data,
            data_quality={
                "quality_score": quality_report["quality_score"],
                "quality_level": quality_report["quality_level"],
                "outliers_detected": any(
                    v for k, v in quality_report["outliers"].items() 
                    if not k.endswith("_z_score") and v
                )
            },
            processing_time_ms=round(processing_time_ms, 2),
            warnings=quality_report["warnings"]
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    
    except Exception as e:
        # ==========================================
        # Error Handling with Monitoring
        # ==========================================
        logger.error(f"Prediction error: {e}")
        
        # Log error to monitor
        monitor = get_monitor()
        monitor.log_error(
            error_type="prediction_error",
            error_message=str(e)
        )
        
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

@app.post('/predict/batch', response_model=BatchPredictResponse, tags=["Prediction"])
def batch_predict(request: BatchListingRequest):
    """
    Batch prediction for multiple listings (max 100)
    
    **Example Request:**
    ```json
    {
        "listings": [
            {
                "neighbourhood_group": "manhattan",
                "room_type": "entire home/apt",
                "minimum_nights": 3,
                "calculated_host_listings_count": 5,
                "availability_365": 200
            },
            {
                "neighbourhood_group": "brooklyn",
                "room_type": "private room",
                "minimum_nights": 1,
                "calculated_host_listings_count": 2,
                "availability_365": 365
            }
        ]
    }
    ```
    """
    global PREDICTION_COUNT
    
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    predictions = []
    
    for listing in request.listings:
        try:
            pred = predict(listing)
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Batch prediction error for listing: {e}")
            predictions.append(PredictResponse(
                price_prediction=0.0,
                confidence="error",
                input_features=listing.model_dump()
            ))
    
    processing_time = (time.time() - start_time) * 1000
    
    return BatchPredictResponse(
        predictions=predictions,
        total_listings=len(predictions),
        processing_time_ms=round(processing_time, 2)
    )

@app.get("/encodings", tags=["Reference"])
def get_encodings() -> Dict:
    """
    Get valid encoding options for categorical features
    
    Returns all valid options for:
    - neighbourhood_group: NYC boroughs
    - room_type: Types of accommodation
    
    Use these values in your prediction requests (case-insensitive)
    """
    return {
        "neighbourhood_group": {
            "options": list(NEIGHBOURHOOD_ENCODING.keys()),
            "codes": NEIGHBOURHOOD_ENCODING,
            "description": "NYC boroughs"
        },
        "room_type": {
            "options": list(ROOM_TYPE_ENCODING.keys()),
            "codes": ROOM_TYPE_ENCODING,
            "description": "Accommodation types"
        }
    }

@app.get("/price-range", tags=["Reference"])
def get_price_range():
    """
    Get typical price ranges by neighbourhood and room type
    
    Note: These are approximate ranges based on training data
    """
    return {
        "manhattan": {
            "entire home/apt": {"min": 80, "max": 500, "average": 180},
            "private room": {"min": 40, "max": 200, "average": 90},
            "shared room": {"min": 20, "max": 100, "average": 50}
        },
        "brooklyn": {
            "entire home/apt": {"min": 60, "max": 300, "average": 120},
            "private room": {"min": 30, "max": 150, "average": 70},
            "shared room": {"min": 15, "max": 80, "average": 40}
        },
        "queens": {
            "entire home/apt": {"min": 50, "max": 250, "average": 100},
            "private room": {"min": 25, "max": 120, "average": 60},
            "shared room": {"min": 15, "max": 70, "average": 35}
        },
        "bronx": {
            "entire home/apt": {"min": 40, "max": 200, "average": 85},
            "private room": {"min": 20, "max": 100, "average": 50},
            "shared room": {"min": 15, "max": 60, "average": 30}
        },
        "staten_island": {
            "entire home/apt": {"min": 45, "max": 220, "average": 90},
            "private room": {"min": 22, "max": 110, "average": 55},
            "shared room": {"min": 15, "max": 65, "average": 32}
        },
        "note": "Actual predictions may vary based on other factors"
    }

if __name__ == "__main__":
    import uvicorn
    # Lấy PORT từ biến môi trường của Render, nếu không có thì dùng 9696
    port = int(os.environ.get("PORT", 9696))
    uvicorn.run(app, host='0.0.0.0', port=port, log_level="info")