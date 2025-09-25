# app/main.py
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import os
from typing import Dict, List

from app.schemas import PredictRequest
from app.model import SkModel, sin_transform, cos_transform

app = FastAPI(
    title="ML Prediction API",
    description="Machine Learning Prediction API for Province Data",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") == "development" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") == "development" else None
)

model = os.getenv("../models/best_rf_model.joblib")
sk = SkModel(model)

# Production CORS configuration
allowed_origins = [
    "https://vanthkrab.com",
    "https://www.vanthkrab.com"
]

# Add localhost only in development
if os.getenv("ENVIRONMENT") == "development":
    allowed_origins.extend(["http://localhost", "http://localhost:3000"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Editable province data - can be modified directly in main.py
def get_province_data() -> List[Dict]:
    """
    Editable method to get province data.
    Modify this method to change province information.
    """
    provinces = [
        {"id": 10, "name": "กรุงเทพมหานคร", "region": "กลาง"},
        {"id": 11, "name": "สมุทรปราการ", "region": "กลาง"},
        {"id": 12, "name": "นนทบุรี", "region": "กลาง"},
        {"id": 13, "name": "ปทุมธานี", "region": "กลาง"},
        {"id": 14, "name": "พระนครศรีอยุธยา", "region": "กลาง"},
        {"id": 15, "name": "อ่างทอง", "region": "กลาง"},
        {"id": 16, "name": "ลพบุรี", "region": "กลาง"},
        {"id": 17, "name": "สิงห์บุรี", "region": "กลาง"},
        {"id": 18, "name": "ชัยนาท", "region": "กลาง"},
        {"id": 19, "name": "สระบุรี", "region": "กลาง"},
        {"id": 20, "name": "ชลบุรี", "region": "ตะวันออก"},
        {"id": 21, "name": "ระยอง", "region": "ตะวันออก"},
        {"id": 22, "name": "จันทบุรี", "region": "ตะวันออก"},
        {"id": 23, "name": "ตราด", "region": "ตะวันออก"},
        {"id": 24, "name": "ฉะเชิงเทรา", "region": "ตะวันออก"},
        {"id": 25, "name": "ปราจีนบุรี", "region": "ตะวันออก"},
        {"id": 26, "name": "นครนายก", "region": "ตะวันออก"},
        {"id": 27, "name": "สระแก้ว", "region": "ตะวันออก"},
        {"id": 30, "name": "นครราชสีมา", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 31, "name": "บุรีรัมย์", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 32, "name": "สุรินทร์", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 33, "name": "ศรีสะเกษ", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 34, "name": "อุบลราชธานี", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 35, "name": "ยโสธร", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 36, "name": "ชัยภูมิ", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 37, "name": "อำนาจเจริญ", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 38, "name": "บึงกาฬ", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 39, "name": "หนองบัวลำภู", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 40, "name": "ขอนแก่น", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 41, "name": "อุดรธานี", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 42, "name": "เลย", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 43, "name": "หนองคาย", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 44, "name": "มหาสารคาม", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 45, "name": "ร้อยเอ็ด", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 46, "name": "กาฬสินธุ์", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 47, "name": "สกลนคร", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 48, "name": "นครพนม", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 49, "name": "มุกดาหาร", "region": "ตะวันออกเฉียงเหนือ"},
        {"id": 50, "name": "เชียงใหม่", "region": "เหนือ"},
        {"id": 51, "name": "ลำพูน", "region": "เหนือ"},
        {"id": 52, "name": "ลำปาง", "region": "เหนือ"},
        {"id": 53, "name": "อุตรดิตถ์", "region": "เหนือ"},
        {"id": 54, "name": "แพร่", "region": "เหนือ"},
        {"id": 55, "name": "น่าน", "region": "เหนือ"},
        {"id": 56, "name": "พะเยา", "region": "เหนือ"},
        {"id": 57, "name": "เชียงราย", "region": "เหนือ"},
        {"id": 58, "name": "แม่ฮ่องสอน", "region": "เหนือ"},
        {"id": 60, "name": "นครสวรรค์", "region": "กลาง"},
        {"id": 61, "name": "อุทัยธานี", "region": "กลาง"},
        {"id": 62, "name": "กำแพงเพชร", "region": "กลาง"},
        {"id": 63, "name": "ตาก", "region": "ตะวันตก"},
        {"id": 64, "name": "สุโขทัย", "region": "กลาง"},
        {"id": 65, "name": "พิษณุโลก", "region": "กลาง"},
        {"id": 66, "name": "พิจิตร", "region": "กลาง"},
        {"id": 67, "name": "เพชรบูรณ์", "region": "กลาง"},
        {"id": 70, "name": "ราชบุรี", "region": "ตะวันตก"},
        {"id": 71, "name": "กาญจนบุรี", "region": "ตะวันตก"},
        {"id": 72, "name": "สุพรรณบุรี", "region": "กลาง"},
        {"id": 73, "name": "นครปฐม", "region": "กลาง"},
        {"id": 74, "name": "สมุทรสาคร", "region": "กลาง"},
        {"id": 75, "name": "สมุทรสงคราม", "region": "กลาง"},
        {"id": 76, "name": "เพชรบุรี", "region": "ตะวันตก"},
        {"id": 77, "name": "ประจวบคีรีขันธ์", "region": "ตะวันตก"},
        {"id": 80, "name": "นครศรีธรรมราช", "region": "ใต้"},
        {"id": 81, "name": "กระบี่", "region": "ใต้"},
        {"id": 82, "name": "พังงา", "region": "ใต้"},
        {"id": 83, "name": "ภูเก็ต", "region": "ใต้"},
        {"id": 84, "name": "สุราษฎร์ธานี", "region": "ใต้"},
        {"id": 85, "name": "ระนอง", "region": "ใต้"},
        {"id": 86, "name": "ชุมพร", "region": "ใต้"},
        {"id": 90, "name": "สงขลา", "region": "ใต้"},
        {"id": 91, "name": "สตูล", "region": "ใต้"},
        {"id": 92, "name": "ตรัง", "region": "ใต้"},
        {"id": 93, "name": "พัทลุง", "region": "ใต้"},
        {"id": 94, "name": "ปัตตานี", "region": "ใต้"},
        {"id": 95, "name": "ยะลา", "region": "ใต้"},
        {"id": 96, "name": "นราธิวาส", "region": "ใต้"},
    ]
    return provinces

# Editable month data - can be modified directly in main.py
def get_month_data() -> List[Dict]:
    """
    Editable method to get month data.
    Modify this method to change month information.
    """
    months = [
        {"value": 1, "name": "มกราคม", "name_en": "January", "season": "หน้าหนาว"},
        {"value": 2, "name": "กุมภาพันธ์", "name_en": "February", "season": "หน้าหนาว"},
        {"value": 3, "name": "มีนาคม", "name_en": "March", "season": "หน้าร้อน"},
        {"value": 4, "name": "เมษายน", "name_en": "April", "season": "หน้าร้อน"},
        {"value": 5, "name": "พฤษภาคม", "name_en": "May", "season": "หน้าร้อน"},
        {"value": 6, "name": "มิถุนายน", "name_en": "June", "season": "หน้าฝน"},
        {"value": 7, "name": "กรกฎาคม", "name_en": "July", "season": "หน้าฝน"},
        {"value": 8, "name": "สิงหาคม", "name_en": "August", "season": "หน้าฝน"},
        {"value": 9, "name": "กันยายน", "name_en": "September", "season": "หน้าฝน"},
        {"value": 10, "name": "ตุลาคม", "name_en": "October", "season": "หน้าฝน"},
        {"value": 11, "name": "พฤศจิกายน", "name_en": "November", "season": "หน้าหนาว"},
        {"value": 12, "name": "ธันวาคม", "name_en": "December", "season": "หน้าหนาว"},
    ]
    return months

# Helper methods for filtering data
def get_province_by_id(province_id: int) -> Dict:
    """Get specific province by ID"""
    provinces = get_province_data()
    for province in provinces:
        if province["id"] == province_id:
            return province
    return {}

def get_provinces_by_region(region: str) -> List[Dict]:
    """Get provinces filtered by region"""
    provinces = get_province_data()
    return [p for p in provinces if p.get("region") == region]

def get_month_by_value(month_value: int) -> Dict:
    """Get specific month by value"""
    months = get_month_data()
    for month in months:
        if month["value"] == month_value:
            return month
    return {}

def get_months_by_season(season: str) -> List[Dict]:
    """Get months filtered by season"""
    months = get_month_data()
    return [m for m in months if m.get("season") == season]

@app.get("/api/")
def read_root():
    return {"message": "Welcome to the ML Prediction API", "version": "1.0.0"}

@app.post("/api/predict")
def predict(req: PredictRequest):
    month_value = req.month_id
    month_sin = sin_transform(month_value)
    month_cos = cos_transform(month_value)
    features = [[req.PROV_ID, month_sin, month_cos]]
    preds, probs = sk.predict(features)

    # Get additional info for response
    province_info = get_province_by_id(req.PROV_ID)
    month_info = get_month_by_value(req.month_id)

    return {
        "prediction": preds[0],
        "probability": probs[0] if probs else None,
        "province": province_info,
        "month": month_info,
        "features_used": {
            "province_id": req.PROV_ID,
            "month_sin": month_sin,
            "month_cos": month_cos
        }
    }

@app.get("/api/provinces")
def get_provinces():
    """Get all provinces with enhanced data"""
    return {"provinces": get_province_data()}

@app.get("/api/provinces/{province_id}")
def get_province(province_id: int):
    """Get specific province by ID"""
    province = get_province_by_id(province_id)
    if not province:
        return {"error": "Province not found", "province_id": province_id}
    return {"province": province}

@app.get("/api/provinces/region/{region}")
def get_provinces_by_region_endpoint(region: str):
    """Get provinces by region"""
    provinces = get_provinces_by_region(region)
    return {"region": region, "provinces": provinces, "count": len(provinces)}

@app.get("/api/months")
def get_months():
    """Get all months with enhanced data"""
    return {"months": get_month_data()}

@app.get("/api/months/{month_value}")
def get_month(month_value: int):
    """Get specific month by value"""
    month = get_month_by_value(month_value)
    if not month:
        return {"error": "Month not found", "month_value": month_value}
    return {"month": month}

@app.get("/api/months/season/{season}")
def get_months_by_season_endpoint(season: str):
    """Get months by season"""
    months = get_months_by_season(season)
    return {"season": season, "months": months, "count": len(months)}

@app.get("/api/features")
def get_features():
    return {"features": sk.get_features()}

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "model_loaded": sk.model is not None
    }

@app.get("/api/info")
def get_api_info():
    """Get API information"""
    return {
        "title": "ML Prediction API",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "endpoints": {
            "predict": "/api/predict",
            "provinces": "/api/provinces",
            "months": "/api/months",
            "health": "/api/health"
        }
    }
