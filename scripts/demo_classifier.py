"""
demo_classifier.py
==================
Backend klasifikasi MODE DEMO menggunakan **MobileNetV2 + bobot ImageNet**
(tanpa perlu training model sendiri).

Tujuan: agar aplikasi Flask bisa langsung "fire up" dan menampilkan hasil
klasifikasi ke 5 kelas sampah, memakai pengetahuan ImageNet yang sudah dilatih
pada jutaan gambar.

Cara kerja:
- MobileNetV2/ImageNet menghasilkan probabilitas untuk 1000 kelas ImageNet.
- Tiap kelas ImageNet dipetakan (via kata kunci) ke salah satu kelas sampah,
  mis. "water_bottle"/"plastic_bag" -> plastik, "carton"/"paper_towel" -> kertas,
  "tin can"/"can_opener" -> logam, "beer_bottle"/"vase" -> kaca, buah/sayur -> organik.
- Probabilitas ImageNet dijumlahkan per kelas sampah lalu dinormalisasi sehingga
  totalnya 100% -> menghasilkan label, confidence, dan distribusi 5 kelas.

CATATAN: ini heuristik untuk DEMO, bukan model final skripsi. Untuk hasil resmi,
tetap latih model sendiri (scripts/train_model.py) pada dataset berlabel.
"""
from __future__ import annotations

import numpy as np

# Urutan kelas mengikuti folder dataset/raw
CLASSES = ["organik", "plastik", "kertas", "logam", "kaca"]

# Pemetaan KATA KUNCI nama kelas ImageNet -> kelas sampah.
# Dicocokkan sebagai substring pada nama kelas ImageNet (huruf kecil).
IMAGENET_HINTS = {
    "plastik": [
        "plastic", "water_bottle", "pop_bottle", "pill_bottle", "lotion",
        "packet", "shopping_bag", "shower_cap", "sunscreen", "soap_dispenser",
        "syringe", "whistle", "toothbrush", "lighter",
    ],
    "kertas": [
        "carton", "paper", "envelope", "toilet_tissue", "menu", "book_jacket",
        "comic_book", "wrapping", "box", "napkin", "binder", "notebook",
        "packet", "carton",
    ],
    "logam": [
        "can", "tin", "metal", "foil", "nail", "screw", "buckle", "spatula",
        "ladle", "frying_pan", "wok", "caldron", "pot", "bottlecap", "chain",
        "padlock", "hook",
    ],
    "kaca": [
        "beer_bottle", "wine_bottle", "beaker", "goblet", "glass", "vase",
        "jar", "cup", "wineglass",
    ],
    "organik": [
        "banana", "orange", "apple", "broccoli", "corn", "mushroom", "lemon",
        "fig", "pineapple", "cabbage", "cucumber", "leaf", "ear", "pomegranate",
        "strawberry", "bell_pepper", "cauliflower", "artichoke", "zucchini",
        "acorn", "hay", "head_cabbage", "butternut_squash",
    ],
}


class DemoClassifier:
    """Klasifikasi gambar ke 5 kelas sampah memakai MobileNetV2/ImageNet."""

    def __init__(self):
        from tensorflow.keras.applications.mobilenet_v2 import (
            MobileNetV2, preprocess_input)
        self._preprocess = preprocess_input
        self.model = MobileNetV2(weights="imagenet")
        self.idx_to_class = self._build_index_map()

    def _build_index_map(self) -> dict:
        """Bangun pemetaan indeks ImageNet (0..999) -> kelas sampah.

        Nama tiap kelas diperoleh dengan men-decode matriks identitas: baris ke-i
        memiliki argmax di indeks i, sehingga decode_predictions mengembalikan nama
        kelas ImageNet untuk indeks i (cara ini lintas-versi Keras/TF).
        """
        from tensorflow.keras.applications.imagenet_utils import decode_predictions

        decoded = decode_predictions(np.eye(1000, dtype="float32"), top=1)
        mapping: dict[int, str] = {}
        for i, entry in enumerate(decoded):
            nm = entry[0][1].lower()  # entry[0] = (wnid, name, score)
            for cls in CLASSES:
                if any(kw in nm for kw in IMAGENET_HINTS[cls]):
                    mapping[i] = cls
                    break
        return mapping

    def predict_array(self, arr_raw: np.ndarray) -> dict:
        """
        arr_raw: array piksel MENTAH [0,255], shape (1, 224, 224, 3)
                 (sama seperti keluaran common.load_image_for_prediction).
        Return dict dengan bentuk yang sama seperti common.predict_array:
            {"index", "label", "confidence", "probabilities"}
        """
        preds = self.model.predict(self._preprocess(arr_raw.copy()), verbose=0)[0]

        scores = {c: 0.0 for c in CLASSES}
        for i, cls in self.idx_to_class.items():
            scores[cls] += float(preds[i])

        total = sum(scores.values())
        if total <= 1e-8:
            # Tidak ada kelas ImageNet yang cocok -> distribusi rata (tidak yakin).
            probs = {c: 1.0 / len(CLASSES) for c in CLASSES}
        else:
            probs = {c: scores[c] / total for c in CLASSES}

        label = max(probs, key=probs.get)
        return {
            "index": CLASSES.index(label),
            "label": label,
            "confidence": probs[label],
            "probabilities": probs,
        }
