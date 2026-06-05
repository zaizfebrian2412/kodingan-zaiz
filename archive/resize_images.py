"""
resize_images.py
================
Perkecil ukuran file gambar JPG/PNG agar maksimal ~1 MB per file, dengan:
  1. menurunkan resolusi bila sisi terpanjang > --max-side (default 1600 px), lalu
  2. menurunkan kualitas JPEG bertahap sampai ukuran file <= --max-mb.

Resolusi 1600 px masih lebih dari cukup karena objek nanti dipotong (crop) dan
model dilatih pada 224x224. Mengedit file DI TEMPAT (overwrite).

Jalankan:
    python scripts/resize_images.py --src dataset/incoming
    python scripts/resize_images.py --src dataset/incoming --max-mb 1 --max-side 1600
"""
import argparse
import io
import sys
from pathlib import Path

from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png"}


def encode_jpeg(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def shrink(path: Path, max_bytes: int, max_side: int) -> tuple[int, int]:
    """Return (ukuran_lama, ukuran_baru) dalam byte. 0,0 bila dilewati."""
    old = path.stat().st_size
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # hormati orientasi EXIF (foto iPhone)
    img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

    # Turunkan kualitas bertahap sampai <= max_bytes
    for q in (90, 85, 80, 75, 70, 65, 60, 55, 50):
        data = encode_jpeg(img, q)
        if len(data) <= max_bytes:
            break
    path.write_bytes(data)
    return old, len(data)


def main() -> None:
    ap = argparse.ArgumentParser(description="Perkecil gambar ke <= ~1 MB.")
    ap.add_argument("--src", default="dataset/incoming", help="Folder gambar.")
    ap.add_argument("--max-mb", type=float, default=1.0, help="Batas ukuran file (MB).")
    ap.add_argument("--max-side", type=int, default=1600, help="Sisi terpanjang maks (px).")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        sys.exit(f"Folder tidak ada: {src}")
    max_bytes = int(args.max_mb * 1024 * 1024)

    imgs = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS)
    if not imgs:
        sys.exit(f"Tidak ada gambar di {src}")

    print(f"{len(imgs)} gambar. Target <= {args.max_mb} MB, sisi maks {args.max_side}px.")
    saved = 0
    over = 0
    for i, p in enumerate(imgs, 1):
        old, new = shrink(p, max_bytes, args.max_side)
        saved += old - new
        if new > max_bytes:
            over += 1
        if i % 50 == 0 or i == len(imgs):
            print(f"  {i}/{len(imgs)} ...")

    print(f"\nSelesai. Hemat ~{saved / 1024 / 1024:.0f} MB.")
    if over:
        print(f"  Catatan: {over} file masih > {args.max_mb} MB (sangat detail). "
              f"Turunkan --max-side bila perlu lebih kecil.")


if __name__ == "__main__":
    main()
