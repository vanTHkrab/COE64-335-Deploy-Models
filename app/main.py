# app/main.py
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.schemas import PredictRequest
from app.model import SkModel

app = FastAPI()

sk = SkModel("models/clf.joblib")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["localhost", "http://localhost:3000", "https://vanthkrab.com", "https://www.vanthkrab.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

#get health check
@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    features = [[req.PROV_ID, req.month_sin, req.month_cos]]
    preds, probs = sk.predict(features)
    return {"pred": preds[0], "prob": probs[0] if probs else None}

@app.get("/features")
def get_features():
    return {"features": sk.get_features()}