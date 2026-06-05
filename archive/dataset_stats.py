"""
dataset_stats.py
================
Hitung jumlah gambar per kelas di dataset/raw/ (dan sisa di dataset/incoming/),
lalu beri penilaian sederhana apakah jumlahnya sudah cukup untuk training.

Jalankan:
    python scripts/dataset_stats.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png"}
# Ambang penilaian (rule of thumb untuk prototype transfer learning)
MIN_OK = 50       # minimal layak coba
GOOD = 150        # cukup baik

BAR_WIDTH = 30


def count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir()
               if p.is_file() and p.suffix.lower() in EXTS)


def status(n: int) -> str:
    if n == 0:
        return "KOSONG"
    if n < MIN_OK:
        return f"kurang (<{MIN_OK})"
    if n < GOOD:
        return f"cukup (>={MIN_OK})"
    return f"baik (>={GOOD})"


def main() -> None:
    raw = common.RAW_DIR
    if not raw.is_dir():
        sys.exit(f"Folder tidak ada: {raw}")

    classes = sorted(p.name for p in raw.iterdir() if p.is_dir())
    counts = {c: count_images(raw / c) for c in classes}
    total = sum(counts.values())
    peak = max(counts.values(), default=0)

    print("=" * 56)
    print(f"Jumlah gambar per kelas di {raw}")
    print("=" * 56)
    for c in classes:
        n = counts[c]
        bar = "#" * round(BAR_WIDTH * n / peak) if peak else ""
        print(f"  {c:<10} {n:>5}  {bar:<{BAR_WIDTH}}  {status(n)}")
    print("-" * 56)
    print(f"  {'TOTAL':<10} {total:>5}  ({len(classes)} kelas)")

    incoming = count_images(common.DATASET_DIR / "incoming")
    if incoming:
        print(f"\n  Belum dilabeli di dataset/incoming/: {incoming} gambar")

    # Ringkasan & peringatan keseimbangan
    print()
    empty = [c for c, n in counts.items() if n == 0]
    low = [c for c, n in counts.items() if 0 < n < MIN_OK]
    if empty:
        print(f"  [!] Kelas masih KOSONG : {', '.join(empty)}")
    if low:
        print(f"  [!] Kelas masih SEDIKIT: {', '.join(low)} (target >= {MIN_OK})")
    nonzero = [n for n in counts.values() if n > 0]
    if len(nonzero) >= 2 and min(nonzero) * 2 < max(nonzero):
        print("  [!] Dataset tidak seimbang (selisih antar kelas > 2x). "
              "Usahakan jumlah tiap kelas mirip.")
    if not empty and not low and total > 0:
        print("  [OK] Semua kelas terisi cukup. Siap: python scripts/split_dataset.py")


if __name__ == "__main__":
    main()
