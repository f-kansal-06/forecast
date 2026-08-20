"""FastAPI service for 24-hour electricity demand forecasts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ARTIFACT_ROOT = Path("outputs")
FEATURE_SCALER_PATH = ARTIFACT_ROOT / "preprocessing" / "feature_scaler.joblib"
TARGET_SCALER_PATH = ARTIFACT_ROOT / "preprocessing" / "target_scaler.joblib"
METADATA_PATH = ARTIFACT_ROOT / "preprocessing" / "preprocessing_metadata.json"
METRICS_PATH = ARTIFACT_ROOT / "evaluation" / "model_comparison.csv"
RUN_METADATA_PATH = ARTIFACT_ROOT / "run_metadata.json"

app = FastAPI(title="BSES Rajdhani Demand Forecast API", version="1.0.0")
ARTIFACT_CACHE: dict[str, Any] = {}


class ForecastRequest(BaseModel):
    """Forecast request containing the latest lookback feature window."""

    records: list[dict[str, Any]] = Field(..., description="Last 168 hours of engineered feature records")


class UpdateModelRequest(BaseModel):
    """Placeholder retraining trigger payload."""

    requested_by: str | None = None
    reason: str | None = None


def load_json(path: Path, default: Any) -> Any:
    """Loads JSON from disk with a default fallback."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model_path() -> Path:
    """Resolves the currently active model artifact from run metadata."""
    run_metadata = load_json(RUN_METADATA_PATH, {})
    model_name = str(run_metadata.get("best_model_name", "bidirectional_lstm"))
    return ARTIFACT_ROOT / "models" / f"{model_name}.keras"


def required_artifacts_exist() -> bool:
    """Checks whether all inference artifacts are available."""
    model_path = resolve_model_path()
    return all(path.exists() for path in [model_path, FEATURE_SCALER_PATH, TARGET_SCALER_PATH, METADATA_PATH])


def load_artifacts() -> dict[str, Any]:
    """Loads and caches the model, scalers, and preprocessing metadata."""
    if ARTIFACT_CACHE:
        return ARTIFACT_CACHE
    if not required_artifacts_exist():
        raise HTTPException(status_code=503, detail="Model artifacts are unavailable. Run python run_all.py first.")
    model_path = resolve_model_path()
    ARTIFACT_CACHE["model"] = tf.keras.models.load_model(model_path)
    ARTIFACT_CACHE["model_path"] = str(model_path)
    ARTIFACT_CACHE["feature_scaler"] = joblib.load(FEATURE_SCALER_PATH)
    ARTIFACT_CACHE["target_scaler"] = joblib.load(TARGET_SCALER_PATH)
    ARTIFACT_CACHE["metadata"] = load_json(METADATA_PATH, {})
    return ARTIFACT_CACHE


def prepare_feature_window(records: list[dict[str, Any]], metadata: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    """Validates and scales the incoming 168-hour engineered feature window."""
    feature_columns = metadata.get("feature_columns", [])
    lookback = int(metadata.get("lookback", 168))
    if len(records) != lookback:
        raise HTTPException(status_code=422, detail=f"Expected exactly {lookback} records")
    frame = pd.DataFrame(records)
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_feature_columns": missing})
    frame = frame.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    numeric = frame[feature_columns].astype(float)
    return numeric.to_numpy(dtype=np.float32), frame


def inverse_forecast(values: np.ndarray, target_scaler: Any) -> np.ndarray:
    """Converts scaled model output to demand kW."""
    return target_scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def build_forecast_response(prediction_kw: np.ndarray, frame: pd.DataFrame) -> dict[str, Any]:
    """Builds the JSON forecast response."""
    last_datetime = None
    if "datetime" in frame.columns:
        parsed = pd.to_datetime(frame["datetime"], errors="coerce")
        if parsed.notna().any():
            last_datetime = parsed.dropna().iloc[-1]
    horizon = []
    for index, value in enumerate(prediction_kw.reshape(-1), start=1):
        item: dict[str, Any] = {"horizon_hour": index, "demand_kw": float(value)}
        if last_datetime is not None:
            item["datetime"] = str(last_datetime + pd.Timedelta(hours=index))
        horizon.append(item)
    return {"forecast_horizon_hours": len(horizon), "forecast": horizon}


@app.post("/forecast")
def forecast(request: ForecastRequest) -> dict[str, Any]:
    """Returns the next 24 hourly demand forecasts."""
    artifacts = load_artifacts()
    feature_window, frame = prepare_feature_window(request.records, artifacts["metadata"])
    scaled = artifacts["feature_scaler"].transform(feature_window)
    model_input = scaled[np.newaxis, :, :]
    prediction_scaled = artifacts["model"].predict(model_input, verbose=0)
    prediction_kw = inverse_forecast(prediction_scaled, artifacts["target_scaler"])
    return build_forecast_response(prediction_kw, frame)


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """Returns model architecture, training metrics, and version metadata."""
    metadata = load_json(METADATA_PATH, {})
    run_metadata = load_json(RUN_METADATA_PATH, {})
    model_path = resolve_model_path()
    metrics: list[dict[str, Any]] = []
    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH).to_dict(orient="records")
    return {
        "model_path": str(model_path),
        "architecture": run_metadata.get("best_family", "unknown"),
        "active_model": run_metadata.get("best_model_name", "bidirectional_lstm"),
        "version": run_metadata.get("total_runtime_seconds", "unversioned"),
        "preprocessing": metadata,
        "metrics": metrics,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Returns service health and artifact readiness."""
    return {"status": "healthy" if required_artifacts_exist() else "degraded", "artifacts_ready": required_artifacts_exist()}


@app.post("/update-model")
def update_model(request: UpdateModelRequest) -> dict[str, Any]:
    """Accepts a placeholder model update trigger."""
    return {
        "status": "accepted",
        "message": "Retraining trigger placeholder recorded for orchestration integration.",
        "requested_by": request.requested_by,
        "reason": request.reason,
    }
