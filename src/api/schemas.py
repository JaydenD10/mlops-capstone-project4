"""
Pydantic Request and Response Schemas for the FastAPI Serving Layer.
Enforces strict input validation and descriptive output structures.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PropertyInput(BaseModel):
    Area_sqft: float = Field(..., ge=100.0, le=10000.0, description="Super built-up area in square feet")
    Bedrooms: int = Field(..., ge=1, le=20, description="Total number of bedrooms")
    Bathrooms: int = Field(..., ge=1, le=20, description="Total number of bathrooms")
    Age_Years: Optional[float] = Field(None, ge=0.0, le=150.0, description="Age of the property in years")
    Distance_to_Center_km: Optional[float] = Field(None, ge=0.0, le=100.0, description="Distance to city centre in km")
    Neighborhood_Score: Optional[float] = Field(None, ge=1.0, le=10.0, description="Neighborhood rating from 1 to 10")
    Has_Parking: int = Field(..., ge=0, le=1, description="1 if parking space is included, 0 otherwise")
    Property_Condition: str = Field(..., description="Condition: Poor, Fair, Good, or Excellent")
    City_Zone: str = Field(..., description="Zone: North, South, East, West, or Central")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Area_sqft": 1250.0,
                "Bedrooms": 3,
                "Bathrooms": 2,
                "Age_Years": 5.0,
                "Distance_to_Center_km": 8.5,
                "Neighborhood_Score": 7.5,
                "Has_Parking": 1,
                "Property_Condition": "Good",
                "City_Zone": "Central"
            }
        }
    }


class BatchPropertyInput(BaseModel):
    properties: List[PropertyInput]


class PredictionOutput(BaseModel):
    predicted_price_lakhs: float
    confidence_interval_low_lakhs: float
    confidence_interval_high_lakhs: float
    model_name: str
    model_version: str
    latency_ms: float
    founder_note: str


class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    total_records: int
    mean_predicted_price_lakhs: float
    batch_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    model_name: str
    model_version: str
    model_loaded: bool
    preprocessor_loaded: bool
    uptime_seconds: float
