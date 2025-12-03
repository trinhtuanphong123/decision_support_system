import pickle
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import xgboost as xgb
import numpy as np


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
            raise ValueError(f"Unknown neighbourhood_group: {v}")
        return NEIGHBOURHOOD_ENCODING[key]

    @field_validator("room_type", mode="before")
    def encode_room_type(cls, v):
        if isinstance(v, int):
            return v
        key = str(v).strip().lower().replace("_", " ")
        if key not in ROOM_TYPE_ENCODING:
            raise ValueError(f"Unknown room_type: {v}")
        return ROOM_TYPE_ENCODING[key]

class PredictResponse(BaseModel):
    price_prediction: float

app = FastAPI(title="house-price-prediction")

with open('model.bin', 'rb') as f_in:
    dv, pipeline = pickle.load(f_in)  # Make sure your model.bin contains both dv and pipeline

@app.post('/predict', response_model=PredictResponse)
def predict(request: ListingRequest):
    X = dv.transform([request.model_dump()])
    features = list(dv.get_feature_names_out())
    dX = xgb.DMatrix(X, feature_names=features)
    y_pred = pipeline.predict(dX)
    price = float(y_pred[0])
    price = round(np.exp(price), 2)
    return PredictResponse(price_prediction=price)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=9696)
