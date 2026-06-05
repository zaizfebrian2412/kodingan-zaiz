"""
evaluate_model.py
=================
Mengevaluasi model terlatih menggunakan dataset test.

Langkah:
1. Load model dari model/waste_classifier.keras
2. Load label dari model/labels.json
3. Baca dataset dari dataset/test
4. Hitung accuracy, precision, recall, F1-score, dan confusion matrix
5. Simpan hasil ke:
   - outputs/classification_report.txt
   - outputs/confusion_matrix.png

Contoh penggunaan:
    python scripts/evaluate_model.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def plot_confusion_matrix(cm, class_names, out_path: Path):
    """Gambar confusion matrix dan simpan sebagai PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Greens")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Label Sebenarnya",
        xlabel="Label Prediksi",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Tulis angka di setiap sel
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix tersimpan -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluasi model klasifikasi sampah.")
    parser.add_argument("--batch-size", type=int, default=common.BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=common.IMG_SIZE[0])
    args = parser.parse_args()

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    common.ensure_dirs()
    img_size = (args.img_size, args.img_size)

    # 1) Load model
    model = common.load_trained_model()
    if model is None:
        print("ERROR: Model belum tersedia. Jalankan training terlebih dahulu "
              "(python scripts/train_model.py).")
        sys.exit(1)

    # 2) Load label
    labels = common.load_labels()
    class_names = [labels[i] for i in sorted(labels.keys())]
    print(f"Kelas: {class_names}")

    # 3) Baca dataset test (urutan dipertahankan, shuffle=False)
    test_ds, ds_class_names = common.make_dataset(
        common.TEST_DIR, img_size=img_size, batch_size=args.batch_size, shuffle=False
    )

    # 4) Kumpulkan label asli & prediksi
    y_true, y_pred = [], []
    for images, batch_labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(batch_labels.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 5) Hitung metrik
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Tampilkan ke terminal
    print("\n" + "=" * 60)
    print("HASIL EVALUASI")
    print("=" * 60)
    print(f"Akurasi (accuracy): {acc * 100:.2f}%\n")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    # Simpan classification report
    report_path = common.OUTPUTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("HASIL EVALUASI MODEL KLASIFIKASI SAMPAH\n")
        f.write("=" * 60 + "\n")
        f.write(f"Jumlah data test : {len(y_true)}\n")
        f.write(f"Akurasi (accuracy): {acc * 100:.2f}%\n\n")
        f.write(report + "\n")
        f.write("Confusion Matrix (baris=asli, kolom=prediksi):\n")
        f.write(f"Urutan kelas: {class_names}\n")
        f.write(np.array2string(cm))
        f.write("\n")
    print(f"\nClassification report tersimpan -> {report_path}")

    # Simpan confusion matrix sebagai gambar
    plot_confusion_matrix(cm, class_names, common.OUTPUTS_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
