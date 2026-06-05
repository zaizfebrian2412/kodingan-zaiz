"""
convert_heic.py
===============
Konversi gambar .HEIC (format foto iPhone) menjadi .JPG agar bisa dipakai
oleh pipeline (split/train/predict/Flask hanya menerima jpg/jpeg/png).

Menggunakan `sips` bawaan macOS -> tidak perlu install library tambahan.

Contoh:
    python scripts/convert_heic.py --src ~/Downloads/dataset_heic --dst dataset/incoming
    python scripts/convert_heic.py --src ~/Downloads/dataset_heic   # dst default: dataset/incoming
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SUPPORTED = {".heic", ".heif"}


def convert_one(src: Path, dst: Path) -> bool:
    """Konversi satu file HEIC -> JPG memakai sips. Return True bila sukses."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["sips", "-s", "format", "jpeg", str(src), "--out", str(dst)],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and dst.exists()


def main() -> None:
    if not shutil.which("sips"):
        sys.exit("ERROR: 'sips' tidak ditemukan. Script ini hanya untuk macOS.")

    ap = argparse.ArgumentParser(description="Konversi HEIC -> JPG (macOS sips).")
    ap.add_argument("--src", required=True, help="Folder berisi file .HEIC (boleh bersarang).")
    ap.add_argument("--dst", default="dataset/incoming",
                    help="Folder tujuan hasil .jpg (default: dataset/incoming).")
    args = ap.parse_args()

    src_dir = Path(args.src).expanduser().resolve()
    dst_dir = Path(args.dst).expanduser().resolve()
    if not src_dir.is_dir():
        sys.exit(f"ERROR: folder sumber tidak ada: {src_dir}")

    heics = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not heics:
        sys.exit(f"Tidak ada file .HEIC di {src_dir}")

    print(f"Ditemukan {len(heics)} file HEIC. Mengonversi -> {dst_dir}")
    ok = 0
    for i, src in enumerate(heics, 1):
        dst = dst_dir / (src.stem + ".jpg")
        # Hindari menimpa: tambahkan suffix bila bentrok
        n = 1
        while dst.exists():
            dst = dst_dir / f"{src.stem}_{n}.jpg"
            n += 1
        if convert_one(src, dst):
            ok += 1
        else:
            print(f"  GAGAL: {src.name}")
        if i % 25 == 0 or i == len(heics):
            print(f"  {i}/{len(heics)} ...")

    print(f"\nSelesai: {ok}/{len(heics)} berhasil dikonversi ke {dst_dir}")
    print("Langkah berikutnya: jalankan tool pelabelan (scripts/label_assist.py) "
          "untuk memotong objek & memberi label ke dataset/raw/<kelas>/.")


if __name__ == "__main__":
    main()
