"""
predict_image.py
================
Memprediksi jenis sampah dari SATU file gambar.

Langkah:
1. Load model & label
2. Preprocess gambar
3. Prediksi kelas
4. Tampilkan nama kelas, confidence score, dan probabilitas semua kelas

Contoh penggunaan:
    python scripts/predict_image.py --image sample.jpg
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Prediksi jenis sampah dari satu gambar.")
    parser.add_argument("--image", required=True, help="Path ke file gambar.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: File gambar tidak ditemukan: {image_path}")
        sys.exit(1)

    # Load model
    model = common.load_trained_model()
    if model is None:
        print("ERROR: Model belum tersedia. Silakan lakukan training terlebih dahulu "
              "(python scripts/train_model.py).")
        sys.exit(1)

    # Load label
    labels = common.load_labels()

    # Preprocess + prediksi
    arr = common.load_image_for_prediction(image_path)
    result = common.predict_array(model, arr, labels)

    # Tampilkan hasil
    print(f"\nGambar    : {image_path}")
    print(f"Prediksi  : {result['label']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print("\nProbabilitas:")
    # Urutkan dari probabilitas tertinggi ke terendah
    for name, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:<12}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
