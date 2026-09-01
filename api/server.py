"""
Network Traffic Forecasting Platform - REST API
================================================
FastAPI server powered by real Kaggle Unicauca network traffic data.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from forecasting.sarimax_engine import NetworkForecastingPlatform
import numpy as np
from datetime import datetime
import uvicorn

app = FastAPI(
    title="Network Traffic Forecasting & KPI Platform",
    description="SARIMAX forecasting powered by Kaggle Unicauca IP Network Traffic dataset (87 Apps)",
    version="2.4.1",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Resolve CSV path — looks in data/ folder at project root
# Override by setting KAGGLE_CSV environment variable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV  = os.path.join(PROJECT_ROOT, "data", "Dataset-Unicauca-Version2-87Atts.csv")
CSV_PATH     = os.environ.get("KAGGLE_CSV", DEFAULT_CSV)

platform = NetworkForecastingPlatform(csv_path=CSV_PATH)


@app.on_event("startup")
async def startup():
    print(f"CSV path   : {CSV_PATH}")
    print(f"CSV exists : {os.path.exists(CSV_PATH)}")
    platform.initialize()


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.4.1", "timestamp": datetime.now().isoformat()}


@app.get("/api/v1/platform/summary")
def get_summary():
    return platform.platform_summary()


@app.get("/api/v1/kpis/current")
def get_current_kpis():
    kpis = platform.get_current_kpis()
    for rec in kpis:
        for k, v in rec.items():
            if isinstance(v, float) and np.isnan(v):
                rec[k] = None
    return {"kpis": kpis, "timestamp": datetime.now().isoformat()}


@app.get("/api/v1/forecast/{site}")
def get_forecast(site: str, hours: int = Query(default=24, ge=1, le=168)):
    try:
        fc = platform.get_site_forecast(site, hours)
        return {"site": site, "horizon_hours": hours, "forecast": fc.to_dict("records")}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/anomalies")
def get_anomalies(hours_back: int = Query(default=24, ge=1, le=720)):
    report = platform.get_anomaly_report(hours_back)
    return {"anomalies": report.to_dict("records"), "count": len(report), "window_hours": hours_back}


@app.get("/api/v1/sites")
def list_sites():
    return {"sites": platform.SITES}


@app.get("/api/v1/models/diagnostics")
def get_diagnostics():
    return {site: model.model_diagnostics() for site, model in platform.models.items()}


@app.get("/api/v1/features/summary")
def get_features():
    return platform.feature_engineer.feature_summary()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)