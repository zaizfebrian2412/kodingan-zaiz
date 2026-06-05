#!/bin/sh
set -e

CERT_DIR=/app/certs
mkdir -p "$CERT_DIR"

if [ ! -f "$CERT_DIR/cert.pem" ]; then
    IP="${SERVER_IP:-127.0.0.1}"
    echo "[entrypoint] Membuat sertifikat self-signed untuk IP: $IP"
    openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
        -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
        -subj "/CN=$IP" -addext "subjectAltName=IP:$IP"
fi

# 1 worker karena TensorFlow boros RAM.
exec gunicorn \
    --chdir app \
    --certfile "$CERT_DIR/cert.pem" \
    --keyfile "$CERT_DIR/key.pem" \
    --bind 0.0.0.0:8443 \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    app:app
