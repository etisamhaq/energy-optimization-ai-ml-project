"""
FastAPI application for Energy Optimization ML API.
"""

from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging
import sys

import pandas as pd
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.predict import EnergyPredictor
from src.optimization.optimizer import EnergyOptimizer
from src.api.security import ObservabilityMiddleware, PREDICTIONS, configured_api_keys, require_api_key
from src.utils.helpers import validate_date_range

# Configure logging
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

MAX_PREDICTION_DAYS = 365
MAX_BUILDINGS_PER_REQUEST = 50

# Initialize components. DATA_DIR and MODELS_DIR point at the pipeline's outputs.
predictor = EnergyPredictor(
    models_dir=os.environ.get("MODELS_DIR", "models"),
    data_dir=os.environ.get("DATA_DIR", "data")
)
optimizer = EnergyOptimizer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload the default model on startup so the first request isn't slow."""
    try:
        logger.info("Preloading models...")
        predictor.load_latest_model("ensemble")
        logger.info("Models preloaded successfully")
    except Exception as e:
        logger.warning(f"Could not preload ensemble model: {e}")
    if not configured_api_keys():
        logger.warning("API_KEYS is not set: authentication is DISABLED. Set it in production.")
    logger.info("Energy Optimization ML API started")
    yield


# Initialize FastAPI app
app = FastAPI(
    title="Energy Optimization ML API",
    description="API for predicting building energy consumption and providing optimization recommendations",
    version="1.0.0",
    lifespan=lifespan,
    # Every endpoint except the public ones (PUBLIC_PATHS) requires an API key when API_KEYS is set
    dependencies=[Depends(require_api_key)]
)

app.add_middleware(
    ObservabilityMiddleware,
    limit_per_minute=int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
)

# Add CORS middleware. Set CORS_ORIGINS to a comma-separated list of allowed origins in production.
cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error_response(e: Exception, context: str) -> HTTPException:
    """Map an exception to an HTTP error with an appropriate status code."""
    if isinstance(e, HTTPException):
        return e
    if isinstance(e, FileNotFoundError):
        logger.error(f"{context}: {e}")
        return HTTPException(status_code=503, detail=f"Model or data not available: {e}")
    if isinstance(e, (ValueError, KeyError)):
        logger.warning(f"{context}: {e}")
        return HTTPException(status_code=400, detail=str(e))
    logger.exception(f"{context}: {e}")
    return HTTPException(status_code=500, detail="Internal server error")


def _validate_dates(start_date: str, end_date: str):
    try:
        return validate_date_range(start_date, end_date, MAX_PREDICTION_DAYS)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    building_id: str = Field(..., description="Building identifier")
    start_date: str = Field(..., description="Start date for prediction (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for prediction (YYYY-MM-DD)")
    model_name: str = Field(default="ensemble", description="Model to use for prediction")
    include_optimization: bool = Field(default=False, description="Include optimization recommendations")
    
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "building_id": "B001",
                "start_date": "2024-03-01",
                "end_date": "2024-03-07",
                "model_name": "ensemble",
                "include_optimization": True
            }
        }
    )


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    building_ids: List[str] = Field(
        ..., min_length=1, max_length=MAX_BUILDINGS_PER_REQUEST, description="List of building identifiers"
    )
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
    analysis_period_days: int = Field(
        default=30, ge=1, le=MAX_PREDICTION_DAYS, description="Days of historical data to analyze"
    )
    electricity_rate: float = Field(default=0.12, gt=0, description="Cost per kWh in dollars")


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
            "health": "/health",
            "readiness": "/ready",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
async def health_check():
    """Liveness: the process is up."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": list(predictor.loaded_models.keys())
    }


@app.get("/ready")
def readiness_check():
    """Readiness: a trained model and its feature definitions can be loaded."""
    try:
        if "ensemble" not in predictor.loaded_models:
            predictor.load_latest_model("ensemble")
        predictor.load_feature_info()
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "not ready", "detail": str(e)})
    return {"status": "ready", "models_loaded": list(predictor.loaded_models.keys())}


@app.get("/metrics")
def metrics():
    """Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict_energy(request: PredictionRequest):
    """Predict energy consumption for a building."""
    try:
        _validate_dates(request.start_date, request.end_date)
        
        # Make predictions
        predictions_df, daily_totals = predictor.predict_range(
            request.building_id, 
            request.start_date, 
            request.end_date,
            request.model_name
        )
        
        # Convert to dict for response
        PREDICTIONS.labels(request.model_name).inc(len(predictions_df))
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
            building_metadata = predictor.data_loader.load_building_metadata([request.building_id])
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
        raise _error_response(e, "Prediction error")


@app.post("/predict/batch")
def batch_predict(request: BatchPredictionRequest):
    """Make predictions for multiple buildings."""
    try:
        _validate_dates(request.start_date, request.end_date)
        results = []
        
        for building_id in request.building_ids:
            predictions_df, daily_totals = predictor.predict_range(
                building_id,
                request.start_date,
                request.end_date,
                request.model_name
            )
            
            PREDICTIONS.labels(request.model_name).inc(len(predictions_df))
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
        raise _error_response(e, "Batch prediction error")


@app.post("/compare")
def compare_models(request: ModelComparisonRequest):
    """Compare predictions from multiple models."""
    try:
        _validate_dates(request.start_date, request.end_date)
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
        raise _error_response(e, "Model comparison error")


@app.post("/optimize")
def optimize_energy(request: OptimizationRequest):
    """Generate optimization recommendations for a building."""
    try:
        # Analyse the most recent measured consumption: recommendations should be
        # based on what the building actually used, not on a forecast
        history = predictor.load_consumption_history(request.building_id).dropna()
        if history.empty:
            raise ValueError(f"No consumption history for building {request.building_id}")
        end = history.index.max()
        recent = history[history.index > end - timedelta(days=request.analysis_period_days)]
        consumption_df = pd.DataFrame({
            'timestamp': recent.index,
            # The optimizer reads this column for both forecast and measured consumption
            'predicted_consumption_kwh': recent.to_numpy()
        })
        
        building_metadata = predictor.data_loader.load_building_metadata([request.building_id])
        metadata_dict = building_metadata.iloc[0].to_dict()
        
        # Create optimizer with custom rate
        custom_optimizer = EnergyOptimizer(electricity_rate=request.electricity_rate)
        report = custom_optimizer.generate_optimization_report(consumption_df, metadata_dict)
        
        return {
            "building_id": request.building_id,
            "analysis_period": {
                "start_date": recent.index.min().strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d")
            },
            "optimization_report": report
        }
        
    except Exception as e:
        raise _error_response(e, "Optimization error")


@app.get("/buildings")
def list_buildings():
    """List available buildings."""
    try:
        building_ids = predictor.data_loader.list_buildings()
        if not building_ids:
            return {"buildings": []}
        metadata = predictor.data_loader.load_building_metadata(building_ids)
        return {"buildings": metadata.to_dict('records')}
        
    except Exception as e:
        raise _error_response(e, "Error listing buildings")


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
            }
        ]
    }


def main():
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host=os.environ.get("API_HOST", "0.0.0.0"), port=int(os.environ.get("API_PORT", "8000")))


if __name__ == "__main__":
    main()