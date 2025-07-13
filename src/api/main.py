"""
FastAPI application for Energy Optimization ML API.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.predict import EnergyPredictor
from src.optimization.optimizer import EnergyOptimizer
from src.data.data_loader import EnergyDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Energy Optimization ML API",
    description="API for predicting building energy consumption and providing optimization recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
predictor = EnergyPredictor()
optimizer = EnergyOptimizer()
data_loader = EnergyDataLoader()

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    building_id: str = Field(..., description="Building identifier")
    start_date: str = Field(..., description="Start date for prediction (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for prediction (YYYY-MM-DD)")
    model_name: str = Field(default="ensemble", description="Model to use for prediction")
    include_optimization: bool = Field(default=False, description="Include optimization recommendations")
    
    class Config:
        schema_extra = {
            "example": {
                "building_id": "B001",
                "start_date": "2024-03-01",
                "end_date": "2024-03-07",
                "model_name": "ensemble",
                "include_optimization": True
            }
        }


class BatchPredictionRequest(BaseModel):
    building_ids: List[str] = Field(..., description="List of building identifiers")
    start_date: str = Field(..., description="Start date for prediction (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for prediction (YYYY-MM-DD)")
    model_name: str = Field(default="ensemble", description="Model to use for prediction")


class BuildingMetadata(BaseModel):
    building_id: str
    building_type: str
    floor_area_sqm: float
    year_built: int
    occupancy_type: str
    hvac_type: str
    energy_star_rating: int


class PredictionResponse(BaseModel):
    building_id: str
    predictions: List[Dict]
    summary: Dict
    optimization_report: Optional[Dict] = None


class ModelComparisonRequest(BaseModel):
    building_id: str
    start_date: str
    end_date: str
    models: List[str] = Field(default=["xgboost", "lightgbm", "random_forest", "ensemble"])


class OptimizationRequest(BaseModel):
    building_id: str
    analysis_period_days: int = Field(default=30, description="Days of historical data to analyze")
    electricity_rate: float = Field(default=0.12, description="Cost per kWh in dollars")


# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Energy Optimization ML API",
        "version": "1.0.0",
        "endpoints": {
            "predictions": "/predict",
            "batch_predictions": "/predict/batch",
            "model_comparison": "/compare",
            "optimization": "/optimize",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": list(predictor.loaded_models.keys())
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_energy(request: PredictionRequest):
    """Predict energy consumption for a building."""
    try:
        # Validate dates
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        if end_date < start_date:
            raise HTTPException(status_code=400, detail="End date must be after start date")
        
        if (end_date - start_date).days > 365:
            raise HTTPException(status_code=400, detail="Prediction period cannot exceed 365 days")
        
        # Make predictions
        predictions_df, daily_totals = predictor.predict_range(
            request.building_id, 
            request.start_date, 
            request.end_date,
            request.model_name
        )
        
        # Convert to dict for response
        predictions = predictions_df.to_dict('records')
        
        # Calculate summary statistics
        summary = {
            "total_predicted_kwh": float(predictions_df['predicted_consumption_kwh'].sum()),
            "average_hourly_kwh": float(predictions_df['predicted_consumption_kwh'].mean()),
            "peak_hourly_kwh": float(predictions_df['predicted_consumption_kwh'].max()),
            "min_hourly_kwh": float(predictions_df['predicted_consumption_kwh'].min()),
            "daily_averages": daily_totals.to_dict('records')
        }
        
        # Generate optimization report if requested
        optimization_report = None
        if request.include_optimization:
            # Get building metadata
            building_metadata = data_loader.load_building_metadata([request.building_id])
            metadata_dict = building_metadata.iloc[0].to_dict()
            
            # Generate optimization report
            opt_report = optimizer.generate_optimization_report(
                predictions_df,
                metadata_dict
            )
            optimization_report = opt_report
        
        return PredictionResponse(
            building_id=request.building_id,
            predictions=predictions,
            summary=summary,
            optimization_report=optimization_report
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Make predictions for multiple buildings."""
    try:
        results = []
        
        for building_id in request.building_ids:
            predictions_df, daily_totals = predictor.predict_range(
                building_id,
                request.start_date,
                request.end_date,
                request.model_name
            )
            
            results.append({
                "building_id": building_id,
                "total_predicted_kwh": float(predictions_df['predicted_consumption_kwh'].sum()),
                "average_daily_kwh": float(daily_totals['daily_total_kwh'].mean())
            })
        
        return {
            "buildings": results,
            "period": {
                "start_date": request.start_date,
                "end_date": request.end_date
            },
            "model": request.model_name
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare")
async def compare_models(request: ModelComparisonRequest):
    """Compare predictions from multiple models."""
    try:
        comparison_df = predictor.compare_models(
            request.building_id,
            request.start_date,
            request.end_date,
            request.models
        )
        
        # Calculate model statistics
        model_stats = {}
        for model in request.models:
            col_name = f'{model}_prediction'
            if col_name in comparison_df.columns:
                model_stats[model] = {
                    "mean": float(comparison_df[col_name].mean()),
                    "std": float(comparison_df[col_name].std()),
                    "total": float(comparison_df[col_name].sum())
                }
        
        return {
            "building_id": request.building_id,
            "period": {
                "start_date": request.start_date,
                "end_date": request.end_date
            },
            "model_statistics": model_stats,
            "ensemble_stats": {
                "mean_prediction": float(comparison_df['mean_prediction'].mean()),
                "prediction_uncertainty": float(comparison_df['std_prediction'].mean())
            }
        }
        
    except Exception as e:
        logger.error(f"Model comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
async def optimize_energy(request: OptimizationRequest):
    """Generate optimization recommendations for a building."""
    try:
        # Get recent predictions for analysis
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.analysis_period_days)
        
        predictions_df, _ = predictor.predict_range(
            request.building_id,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            "ensemble"
        )
        
        # Get building metadata
        building_metadata = data_loader.load_building_metadata([request.building_id])
        metadata_dict = building_metadata.iloc[0].to_dict()
        
        # Create optimizer with custom rate
        custom_optimizer = EnergyOptimizer(electricity_rate=request.electricity_rate)
        
        # Generate optimization report
        report = custom_optimizer.generate_optimization_report(
            predictions_df,
            metadata_dict
        )
        
        return {
            "building_id": request.building_id,
            "analysis_period": {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d")
            },
            "optimization_report": report
        }
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/buildings")
async def list_buildings():
    """List available buildings."""
    try:
        # Get sample building IDs
        building_ids = [f"B{str(i).zfill(3)}" for i in range(1, 11)]
        buildings = []
        
        for building_id in building_ids:
            metadata = data_loader.load_building_metadata([building_id])
            if not metadata.empty:
                buildings.append(metadata.iloc[0].to_dict())
        
        return {"buildings": buildings}
        
    except Exception as e:
        logger.error(f"Error listing buildings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {
                "name": "ensemble",
                "description": "Ensemble model combining multiple algorithms",
                "type": "ensemble"
            },
            {
                "name": "xgboost",
                "description": "XGBoost gradient boosting model",
                "type": "ml"
            },
            {
                "name": "lightgbm",
                "description": "LightGBM gradient boosting model",
                "type": "ml"
            },
            {
                "name": "random_forest",
                "description": "Random Forest model",
                "type": "ml"
            },
            {
                "name": "prophet",
                "description": "Facebook Prophet time series model",
                "type": "time_series"
            },
            {
                "name": "arima",
                "description": "ARIMA time series model",
                "type": "time_series"
            },
            {
                "name": "lstm",
                "description": "LSTM deep learning model",
                "type": "deep_learning"
            }
        ]
    }


# Background task for model preloading
async def preload_models():
    """Preload models in background."""
    try:
        logger.info("Preloading models...")
        predictor.load_latest_model("ensemble")
        logger.info("Models preloaded successfully")
    except Exception as e:
        logger.error(f"Error preloading models: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    # Schedule model preloading
    background_tasks = BackgroundTasks()
    background_tasks.add_task(preload_models)
    
    logger.info("Energy Optimization ML API started")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)