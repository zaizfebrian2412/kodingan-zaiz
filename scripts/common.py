"""
common.py
=========
Modul bantu (helper) yang dipakai ulang oleh beberapa script:
- train_model.py
- evaluate_model.py
- predict_image.py
- app/app.py (aplikasi Flask)

Tujuannya agar konfigurasi (ukuran gambar, path, dll) dan logika
preprocessing/prediksi hanya ditulis di SATU tempat (single source of truth),
sehingga konsisten antara proses training dan inference.

Catatan penting tentang preprocessing
--------------------------------------
Preprocessing MobileNetV2 (mengubah piksel [0, 255] menjadi [-1, 1]) DIBAKAR
ke dalam model sebagai layer. Akibatnya, saat prediksi kita cukup memberikan
piksel RGB mentah dengan shape (1, 224, 224, 3) dan model menangani normalisasi
secara internal. Ini mencegah bug "double preprocessing" yang umum terjadi.

Layer Rescaling(scale=1/127.5, offset=-1.0) secara matematis identik dengan
tensorflow.keras.applications.mobilenet_v2.preprocess_input (mode "tf"),
namun lebih aman saat menyimpan/memuat model format ".keras".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Path penting (relatif terhadap root project, aman lintas sistem operasi)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "waste_classifier.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOADS_DIR = PROJECT_ROOT / "app" / "static" / "uploads"

# ----------------------------------------------------------------------
# Konfigurasi default model
# ----------------------------------------------------------------------
IMG_SIZE = (224, 224)          # (tinggi, lebar) untuk MobileNetV2
CHANNELS = 3                   # RGB
BATCH_SIZE = 32
SEED = 42

# Ekstensi gambar yang diizinkan (untuk upload Flask & split dataset)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


# ======================================================================
# Util folder & label
# ======================================================================
def ensure_dirs() -> None:
    """Buat semua folder yang dibutuhkan bila belum ada (aman dipanggil berkali-kali)."""
    for d in (TRAIN_DIR, VAL_DIR, TEST_DIR, MODEL_DIR, OUTPUTS_DIR, UPLOADS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_labels(class_names, path: Path = LABELS_PATH) -> dict:
    """
    Simpan daftar kelas ke labels.json dengan format {"0": "organik", ...}.
    `class_names` adalah list nama kelas terurut sesuai indeks model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping = {str(i): name for i, name in enumerate(class_names)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    return mapping


def load_labels(path: Path = LABELS_PATH) -> dict:
    """
    Baca labels.json dan kembalikan dict {0: "organik", 1: "plastik", ...}
    (kunci berupa integer agar mudah dipasangkan dengan indeks prediksi).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"File label tidak ditemukan: {path}\n"
            "Jalankan training terlebih dahulu (python scripts/train_model.py)."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ======================================================================
# Preprocessing & prediksi gambar (inference)
# ======================================================================
def load_image_for_prediction(image_path, img_size=IMG_SIZE) -> np.ndarray:
    """
    Baca satu file gambar lalu ubah menjadi tensor siap prediksi.

    Langkah:
    1. Buka gambar
    2. Convert ke RGB (buang alpha / grayscale)
    3. Resize ke 224 x 224
    4. Ubah ke array float32 dengan nilai piksel mentah [0, 255]
    5. Tambah dimensi batch -> shape (1, 224, 224, 3)

    Normalisasi MobileNetV2 dilakukan di dalam model, jadi tidak dilakukan di sini.
    """
    from PIL import Image

    # img_size = (tinggi, lebar); PIL.resize butuh (lebar, tinggi)
    target = (img_size[1], img_size[0])
    img = Image.open(image_path).convert("RGB").resize(target)
    arr = np.asarray(img, dtype="float32")          # (224, 224, 3)
    arr = np.expand_dims(arr, axis=0)               # (1, 224, 224, 3)
    return arr


def predict_array(model, arr: np.ndarray, labels: dict) -> dict:
    """
    Jalankan prediksi pada array gambar (1, 224, 224, 3) dan kembalikan ringkasan:
    {
      "index": 1,
      "label": "plastik",
      "confidence": 0.9245,
      "probabilities": {"organik": 0.012, "plastik": 0.9245, ...}
    }
    """
    preds = model.predict(arr, verbose=0)[0]        # vektor probabilitas (num_classes,)
    idx = int(np.argmax(preds))
    label = labels.get(idx, str(idx))
    confidence = float(preds[idx])
    probabilities = {
        labels.get(i, str(i)): float(p) for i, p in enumerate(preds)
    }
    return {
        "index": idx,
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def load_trained_model(path: Path = MODEL_PATH):
    path = Path(path)
    if not path.exists():
        return None
    import tensorflow as tf
    weights_path = path.parent / "waste_classifier_weights.weights.h5"
    model = build_model(num_classes=5)
    model.load_weights(str(weights_path))
    return model


# ======================================================================
# Dataset & arsitektur model (dipakai saat training/evaluasi)
# ======================================================================
def make_dataset(directory, img_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False):
    """
    Buat tf.data.Dataset dari folder berstruktur:
        directory/<nama_kelas>/<gambar>.jpg

    Mengembalikan (dataset, class_names). Label berupa integer (label_mode="int")
    agar cocok dengan loss sparse_categorical_crossentropy.
    """
    import tensorflow as tf

    directory = Path(directory)
    if not directory.exists() or not any(directory.iterdir()):
        raise FileNotFoundError(
            f"Folder dataset kosong / tidak ada: {directory}\n"
            "Pastikan Anda sudah menjalankan: python scripts/split_dataset.py"
        )

    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED,
    )
    class_names = list(ds.class_names)              # ambil sebelum prefetch
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, class_names


def build_data_augmentation():
    """
    Bangun blok data augmentation (hanya aktif saat training).
    Teknik: flip horizontal, rotasi, zoom, translasi, dan kontras acak.
    """
    from tensorflow.keras import layers, Sequential

    return Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.15),
        ],
        name="data_augmentation",
    )


def build_model(num_classes: int, img_size=IMG_SIZE, base_trainable: bool = False):
    """
    Bangun model MobileNetV2 transfer learning.

    Arsitektur:
        Input (224,224,3)
        -> Data Augmentation (aktif saat training saja)
        -> Rescaling  (setara mobilenet_v2.preprocess_input)
        -> MobileNetV2 (include_top=False, bobot ImageNet, frozen)
        -> GlobalAveragePooling2D
        -> Dropout 0.3
        -> Dense 128 (ReLU)
        -> Dropout 0.3
        -> Dense num_classes (softmax)
    """
    from tensorflow.keras import Input, layers, Model
    from tensorflow.keras.applications import MobileNetV2

    inputs = Input(shape=(img_size[0], img_size[1], CHANNELS))

    # 1) Augmentasi (otomatis non-aktif saat inference/prediksi)
    x = build_data_augmentation()(inputs)

    # 2) Normalisasi MobileNetV2: [0,255] -> [-1,1]  (= preprocess_input mode "tf")
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(x)

    # 3) Base model pretrained
    base_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], CHANNELS),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = base_trainable
    x = base_model(x, training=False)

    # 4) Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="waste_classifier")
    return model
