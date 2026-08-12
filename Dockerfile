# ── Stage 1: Base Image ───────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set env variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Install Python Dependencies ─────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi==0.111.0 \
        uvicorn==0.29.0 \
        pydantic==2.7.1 \
        scikit-learn==1.4.2 \
        xgboost==2.0.3 \
        lightgbm==4.3.0 \
        joblib==1.4.0 \
        pandas==2.2.2 \
        numpy==1.26.4 \
        shap==0.45.0

# ── Stage 3: Copy App Files ───────────────────────────────────────────────────
COPY api/        ./api/
COPY models/     ./models/
COPY data/processed/ ./data/processed/

# ── Stage 4: Expose & Run ─────────────────────────────────────────────────────
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
