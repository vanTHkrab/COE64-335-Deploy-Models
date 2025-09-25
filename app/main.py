# app/main.py
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import os

from app.schemas import PredictRequest
from app.model import SkModel, sin_transform, cos_transform
from app.province import get_all_provinces, get_all_months

app = FastAPI()

model = os.getenv("MODEL_PATH", "../models/best_rf_model.joblib")
sk = SkModel(model)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["localhost", "http://localhost:3000", "https://vanthkrab.com", "https://www.vanthkrab.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@app.get("/api/")
def read_root():
    return {"message": "Welcome to the Prediction API"}

@app.post("/api/predict")
def predict(req: PredictRequest):
    month_value = req.month_id
    month_sin = sin_transform(month_value)
    month_cos = cos_transform(month_value)
    features = [[req.PROV_ID, month_sin, month_cos]]
    preds, probs = sk.predict(features)
    return {"pred": preds[0], "prob": probs[0] if probs else None}

@app.get("/api/provinces")
def get_provinces():
    return {"provinces": get_all_provinces()}

@app.get("/api/months")
def get_months():
    return {"months": get_all_months()}

@app.get("/api/features")
def get_features():
    return {"features": sk.get_features()}

@app.get("/api/health")
def health_check():
    return {"status": "ok"}