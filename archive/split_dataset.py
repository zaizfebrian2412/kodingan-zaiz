"""
split_dataset.py
================
Membagi dataset mentah (dataset/raw/<kelas>/) menjadi tiga subset:
    - dataset/train/<kelas>/
    - dataset/validation/<kelas>/
    - dataset/test/<kelas>/

Pembagian dilakukan PER KELAS agar distribusi tiap kategori tetap seimbang.
Rasio default: 70% train, 15% validation, 15% test.

Contoh penggunaan:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --ratios 0.8 0.1 0.1 --seed 7
    python scripts/split_dataset.py --move   # pindahkan file (bukan menyalin)
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

# Pastikan modul common dapat di-import baik dari root maupun dari folder scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def list_classes(raw_dir: Path):
    """Ambil daftar folder kelas (subfolder) di dalam dataset/raw."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Folder dataset mentah tidak ditemukan: {raw_dir}")
    classes = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(
            f"Tidak ada subfolder kelas di {raw_dir}.\n"
            "Letakkan gambar pada dataset/raw/<nama_kelas>/ terlebih dahulu."
        )
    return classes


def list_images(class_dir: Path):
    """Ambil semua file gambar yang valid di dalam satu folder kelas."""
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in common.IMAGE_EXTENSIONS
    )


def split_indices(n: int, ratios):
    """Hitung jumlah item untuk train/val/test berdasarkan rasio."""
    train_ratio, val_ratio, _ = ratios
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val   # sisanya untuk test agar total selalu pas
    return n_train, n_val, n_test


def transfer(files, dest_dir: Path, move: bool):
    """Salin (atau pindahkan) daftar file ke folder tujuan."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = dest_dir / f.name
        if move:
            shutil.move(str(f), str(target))
        else:
            shutil.copy2(str(f), str(target))


def main():
    parser = argparse.ArgumentParser(description="Split dataset menjadi train/validation/test.")
    parser.add_argument("--source", default=str(common.RAW_DIR),
                        help="Folder dataset mentah (default: dataset/raw).")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Rasio train val test (default: 0.7 0.15 0.15).")
    parser.add_argument("--seed", type=int, default=common.SEED,
                        help="Seed acak agar pembagian dapat direproduksi.")
    parser.add_argument("--move", action="store_true",
                        help="Pindahkan file alih-alih menyalin.")
    parser.add_argument("--clean", action="store_true",
                        help="Kosongkan folder train/validation/test sebelum split.")
    args = parser.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        parser.error(f"Jumlah rasio harus 1.0, sekarang {sum(args.ratios)}")

    raw_dir = Path(args.source)
    random.seed(args.seed)

    # Bersihkan folder hasil split lama bila diminta
    if args.clean:
        for d in (common.TRAIN_DIR, common.VAL_DIR, common.TEST_DIR):
            if d.exists():
                shutil.rmtree(d)

    classes = list_classes(raw_dir)
    print(f"Ditemukan {len(classes)} kelas: {', '.join(classes)}")
    print(f"Rasio  -> train:{args.ratios[0]}  val:{args.ratios[1]}  test:{args.ratios[2]}")
    print("-" * 60)

    total = {"train": 0, "validation": 0, "test": 0}

    for cls in classes:
        images = list_images(raw_dir / cls)
        if not images:
            print(f"  [!] Kelas '{cls}' kosong, dilewati.")
            continue

        random.shuffle(images)
        n_train, n_val, n_test = split_indices(len(images), args.ratios)

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        transfer(train_files, common.TRAIN_DIR / cls, args.move)
        transfer(val_files, common.VAL_DIR / cls, args.move)
        transfer(test_files, common.TEST_DIR / cls, args.move)

        total["train"] += len(train_files)
        total["validation"] += len(val_files)
        total["test"] += len(test_files)

        print(f"  {cls:<12} total={len(images):<5} "
              f"train={len(train_files):<5} val={len(val_files):<5} test={len(test_files)}")

    print("-" * 60)
    print(f"Selesai. Total -> train:{total['train']}  "
          f"validation:{total['validation']}  test:{total['test']}")
    print(f"Hasil split tersimpan di: {common.DATASET_DIR}")


if __name__ == "__main__":
    main()
