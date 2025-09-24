# ========== Base ==========
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# runtime libs ที่ numpy/scipy/scikit-learn ต้องใช้ (ไม่ต้องคอมไพล์)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# ติดตั้งไลบรารี
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# คัดซอร์สโค้ด
COPY app ./app
# คัดโมเดล (ถ้าอยาก mount ตอนรันก็ข้าม COPY นี้ได้)
COPY models ./models

# ========== Production ==========
FROM base AS prod
ENV MODEL_PATH=/app/models/clf.joblib
EXPOSE 8000
# Gunicorn + UvicornWorker เหมาะกับ production
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", \
     "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "60"]
