"""
app.py
======
Aplikasi web Flask (prototype) untuk klasifikasi citra sampah.

Alur:
1. User membuka halaman utama (form upload).
2. User mengunggah foto sampah (jpg/jpeg/png).
3. Gambar disimpan, lalu diproses & diprediksi oleh model CNN.
4. Hasil ditampilkan: label, confidence, dan tabel probabilitas semua kelas.

Catatan:
- Model dimuat SATU KALI saat aplikasi dijalankan (efisien, tanpa training ulang).
- Bila model belum tersedia, aplikasi tetap berjalan dan menampilkan pesan jelas.

Menjalankan:
    python app/app.py
    lalu buka http://127.0.0.1:5000
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

# Tambahkan folder scripts/ ke path agar bisa memakai ulang modul common
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import common  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # batas upload 10 MB

# Pastikan folder upload tersedia
common.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Muat model & label SATU KALI saat startup
# --------------------------------------------------------------------
# Mode demo: pakai MobileNetV2/ImageNet bila model terlatih belum ada,
# atau bila variabel lingkungan WASTE_DEMO=1 dipaksa aktif.
FORCE_DEMO = os.environ.get("WASTE_DEMO", "").lower() in ("1", "true", "yes")

MODEL = None if FORCE_DEMO else common.load_trained_model()
try:
    LABELS = common.load_labels() if MODEL is not None else None
except FileNotFoundError:
    LABELS = None

DEMO = None
DEMO_MODE = False

if MODEL is not None and LABELS is not None:
    MODEL_READY = True
    print(f"[INFO] Model dimuat. Kelas: {list(LABELS.values())}")
else:
    # Fallback backend klasifikasi (lihat scripts/demo_classifier.py)
    from demo_classifier import DemoClassifier, CLASSES as DEMO_CLASSES  # noqa: E402
    print("[INFO] Memuat model klasifikasi ...")
    DEMO = DemoClassifier()
    DEMO_MODE = True
    MODEL_READY = True
    print(f"[INFO] Model siap. Kelas: {DEMO_CLASSES}")

MODEL_NOT_READY_MSG = "Model belum tersedia. Silakan lakukan training terlebih dahulu."


def allowed_file(filename: str) -> bool:
    """Cek ekstensi file yang diizinkan (jpg/jpeg/png)."""
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in common.ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html", model_ready=MODEL_READY, demo=DEMO_MODE,
                           model_message=None if MODEL_READY else MODEL_NOT_READY_MSG)


@app.route("/predict", methods=["POST"])
def predict():
    # Validasi: model tersedia?
    if not MODEL_READY:
        return render_template("index.html", model_ready=False, demo=DEMO_MODE,
                               model_message=MODEL_NOT_READY_MSG), 503

    # Validasi: file ada?
    if "image" not in request.files or request.files["image"].filename == "":
        return render_template("index.html", model_ready=True, demo=DEMO_MODE,
                               error="Silakan pilih file gambar terlebih dahulu."), 400

    file = request.files["image"]

    # Validasi: ekstensi benar?
    if not allowed_file(file.filename):
        return render_template("index.html", model_ready=True, demo=DEMO_MODE,
                               error="Format tidak didukung. Gunakan jpg, jpeg, atau png."), 400

    # Simpan file dengan nama unik agar tidak saling menimpa
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{secure_filename(file.filename)}"
    save_path = common.UPLOADS_DIR / filename
    file.save(str(save_path))

    # Preprocess + prediksi
    try:
        arr = common.load_image_for_prediction(save_path)
        if DEMO_MODE:
            result = DEMO.predict_array(arr)          # MobileNetV2/ImageNet
        else:
            result = common.predict_array(MODEL, arr, LABELS)  # model terlatih
    except Exception as exc:  # noqa: BLE001 - tampilkan error secara ramah
        return render_template("index.html", model_ready=True, demo=DEMO_MODE,
                               error=f"Gagal memproses gambar: {exc}"), 500

    # Siapkan data untuk template (urut dari probabilitas tertinggi)
    probabilities = sorted(
        ({"name": name, "percent": prob * 100}
         for name, prob in result["probabilities"].items()),
        key=lambda x: x["percent"], reverse=True,
    )

    return render_template(
        "result.html",
        image_url=url_for("static", filename=f"uploads/{filename}"),
        label=result["label"],
        confidence=result["confidence"] * 100,
        probabilities=probabilities,
        demo=DEMO_MODE,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
