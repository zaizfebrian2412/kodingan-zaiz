# =====================================================================
# Image untuk aplikasi web Klasifikasi Citra Sampah (mode demo/inference).
# Menjalankan gunicorn LANGSUNG dengan TLS (self-signed) -> HTTPS via IP.
# =====================================================================
FROM python:3.11-slim

# openssl untuk membuat sertifikat self-signed saat container start.
RUN apt-get update && apt-get install -y --no-install-recommends openssl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WASTE_DEMO=1

WORKDIR /app

# 1) Install dependency dulu (layer cache).
COPY requirements-deploy.txt .
RUN pip install --upgrade pip && pip install -r requirements-deploy.txt

# 2) Salin kode aplikasi.
COPY app/ ./app/
COPY scripts/common.py scripts/demo_classifier.py ./scripts/
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# 3) Pra-unduh bobot MobileNetV2 + index ImageNet ke dalam image,
#    supaya start cepat & tidak tergantung internet saat runtime.
RUN python -c "import sys; sys.path.insert(0,'scripts'); \
from demo_classifier import DemoClassifier; DemoClassifier(); print('weights cached')"

EXPOSE 8443
ENTRYPOINT ["./docker-entrypoint.sh"]
