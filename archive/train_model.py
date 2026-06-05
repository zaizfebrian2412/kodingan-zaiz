"""
train_model.py
==============
Melatih model klasifikasi sampah berbasis MobileNetV2 (transfer learning)
dengan data augmentation.

Langkah:
1. Baca dataset dari dataset/train dan dataset/validation.
2. Ambil nama kelas otomatis lalu simpan ke model/labels.json.
3. Bangun & latih model MobileNetV2.
4. Simpan model terbaik ke model/waste_classifier.keras (lewat ModelCheckpoint).
5. Simpan grafik akurasi & loss ke folder outputs/.
6. Tampilkan ringkasan hasil training.

Contoh penggunaan:
    python scripts/train_model.py
    python scripts/train_model.py --epochs 30 --batch-size 16 --lr 0.0001
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def plot_history(history, outputs_dir: Path):
    """Simpan grafik accuracy dan loss training/validation ke folder outputs."""
    import matplotlib
    matplotlib.use("Agg")  # backend non-interaktif (aman tanpa GUI)
    import matplotlib.pyplot as plt

    outputs_dir.mkdir(parents=True, exist_ok=True)
    hist = history.history
    epochs_range = range(1, len(hist["loss"]) + 1)

    # --- Grafik Accuracy ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, hist.get("accuracy", []), marker="o", label="Train Accuracy")
    plt.plot(epochs_range, hist.get("val_accuracy", []), marker="o", label="Validation Accuracy")
    plt.title("Akurasi Training & Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    acc_path = outputs_dir / "training_accuracy.png"
    plt.savefig(acc_path, bbox_inches="tight", dpi=120)
    plt.close()

    # --- Grafik Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, hist.get("loss", []), marker="o", label="Train Loss")
    plt.plot(epochs_range, hist.get("val_loss", []), marker="o", label="Validation Loss")
    plt.title("Loss Training & Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_path = outputs_dir / "training_loss.png"
    plt.savefig(loss_path, bbox_inches="tight", dpi=120)
    plt.close()

    print(f"Grafik akurasi  -> {acc_path}")
    print(f"Grafik loss     -> {loss_path}")


def main():
    parser = argparse.ArgumentParser(description="Latih model MobileNetV2 klasifikasi sampah.")
    parser.add_argument("--epochs", type=int, default=25, help="Jumlah epoch (default: 25).")
    parser.add_argument("--batch-size", type=int, default=common.BATCH_SIZE,
                        help="Ukuran batch (default: 32).")
    parser.add_argument("--img-size", type=int, default=common.IMG_SIZE[0],
                        help="Ukuran sisi gambar persegi (default: 224).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate awal (default: 0.0001).")
    args = parser.parse_args()

    # Import TensorFlow di sini supaya pesan error lebih jelas bila belum terinstall
    import tensorflow as tf

    common.ensure_dirs()
    img_size = (args.img_size, args.img_size)

    print("Memuat dataset training & validation...")
    train_ds, class_names = common.make_dataset(
        common.TRAIN_DIR, img_size=img_size, batch_size=args.batch_size, shuffle=True
    )
    val_ds, _ = common.make_dataset(
        common.VAL_DIR, img_size=img_size, batch_size=args.batch_size, shuffle=False
    )

    num_classes = len(class_names)
    print(f"Jumlah kelas: {num_classes} -> {class_names}")

    # Simpan labels.json (sumber kebenaran nama kelas untuk evaluasi & inference)
    common.save_labels(class_names)
    print(f"Label disimpan ke: {common.LABELS_PATH}")

    print("Membangun model MobileNetV2...")
    model = common.build_model(num_classes, img_size=img_size, base_trainable=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",   # label berupa integer
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(common.MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print(f"\nMulai training ({args.epochs} epoch)...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Pastikan model tersimpan walau ModelCheckpoint belum sempat menulis
    if not common.MODEL_PATH.exists():
        model.save(common.MODEL_PATH)
    print(f"\nModel terbaik tersimpan di: {common.MODEL_PATH}")

    plot_history(history, common.OUTPUTS_DIR)

    # Ringkasan hasil training
    val_acc = history.history.get("val_accuracy", [0])
    val_loss = history.history.get("val_loss", [0])
    best_idx = int(max(range(len(val_acc)), key=lambda i: val_acc[i])) if val_acc else 0
    print("\n" + "=" * 60)
    print("RINGKASAN TRAINING")
    print("=" * 60)
    print(f"Epoch terbaik (val_accuracy) : {best_idx + 1}")
    print(f"Validation accuracy terbaik  : {val_acc[best_idx] * 100:.2f}%")
    print(f"Validation loss pada epoch tsb: {val_loss[best_idx]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
