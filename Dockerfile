FROM python:3.11-slim

WORKDIR /app

# System libs: kept minimal since geopandas 1.1.3's manylinux wheel bundles
# GDAL/GEOS/PROJ. libgomp1 is included as a safety net for numpy/matplotlib's
# OpenMP dependency, which some slim base images lack.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT (defaults to 8080) — the app must listen on it.
ENV PORT=8080
EXPOSE 8080

CMD streamlit run app_consolidated.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false