"""
label_assist.py
================
Tool pelabelan semi-otomatis untuk dataset sampah yang BELUM dilabeli dan berisi
banyak objek per foto (mixed scene).

Alur per foto:
  1. Foto ditampilkan.
  2. Anda men-drag kotak (mouse kiri) mengelilingi SATU objek sampah.
  3. AI (MobileNetV2 + ImageNet, model yang sudah dipakai project ini) menebak
     kelasnya -> ditampilkan sebagai SARAN di judul jendela.
  4. Anda tekan angka 1-5 untuk menyimpan crop ke dataset/raw/<kelas>/
     (tekan Enter = terima saran AI). Ulangi untuk objek lain di foto yang sama.

Tombol:
  drag mouse kiri : gambar kotak crop
  1..5            : simpan crop ke kelas tsb (lihat legenda di judul)
  Enter / Space   : simpan crop memakai SARAN AI
  u               : undo crop terakhir untuk foto ini
  n / ->          : foto berikutnya
  p / <-          : foto sebelumnya
  q / Esc         : keluar

Jalankan:
    python scripts/label_assist.py --src dataset/incoming
    python scripts/label_assist.py --src dataset/incoming --no-ai   # tanpa saran AI
"""
import argparse
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# Kelas mengikuti folder dataset/raw (urut tetap 1..5 untuk hotkey)
CLASSES = ["organik", "plastik", "kertas", "logam", "kaca"]

# Pemetaan KATA KUNCI label ImageNet -> kelas sampah (heuristik, untuk SARAN saja).
# Saran AI tidak harus benar — Anda yang memverifikasi.
IMAGENET_HINTS = {
    "plastik": ["plastic", "water_bottle", "pop_bottle", "pill_bottle", "lotion",
                "packet", "shopping_bag", "shower_cap", "sunscreen", "soap_dispenser"],
    "kertas": ["carton", "paper", "envelope", "toilet_tissue", "menu", "book_jacket",
               "comic_book", "wrapping", "box"],
    "logam": ["can", "tin", "metal", "foil", "nail", "screw", "lighter", "buckle",
              "spatula", "ladle"],
    "kaca": ["beer_bottle", "wine_bottle", "beaker", "goblet", "glass", "vase", "jar"],
    "organik": ["banana", "orange", "apple", "broccoli", "corn", "mushroom", "lemon",
                "fig", "pineapple", "cabbage", "cucumber", "leaf", "ear", "pomegranate"],
}


class AISuggester:
    """Pembungkus MobileNetV2/ImageNet untuk menyarankan kelas dari sebuah crop."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.model = None
        if not enabled:
            return
        try:
            import numpy as np  # noqa
            from tensorflow.keras.applications.mobilenet_v2 import (
                MobileNetV2, preprocess_input, decode_predictions)
            self._np = __import__("numpy")
            self._pre = preprocess_input
            self._decode = decode_predictions
            print("Memuat model saran AI (MobileNetV2/ImageNet) ...")
            self.model = MobileNetV2(weights="imagenet")
            print("Model saran AI siap.")
        except Exception as e:  # offline / tanpa bobot -> nonaktif diam-diam
            print(f"(Saran AI dinonaktifkan: {e})")
            self.enabled = False

    def suggest(self, pil_crop: Image.Image):
        """Return (kelas_saran | None, teks_debug)."""
        if not self.enabled or self.model is None:
            return None, "AI off"
        img = pil_crop.convert("RGB").resize((224, 224))
        arr = self._np.expand_dims(self._np.array(img, dtype="float32"), 0)
        preds = self.model.predict(self._pre(arr), verbose=0)
        top = self._decode(preds, top=5)[0]  # [(id, name, prob), ...]
        names = [n.lower() for (_, n, _) in top]
        for cls in CLASSES:  # cari kecocokan kata kunci pada top-5
            for kw in IMAGENET_HINTS[cls]:
                if any(kw in n for n in names):
                    return cls, f"{top[0][1]} ({top[0][2]*100:.0f}%)"
        return None, f"{top[0][1]} ({top[0][2]*100:.0f}%)"


class Labeler:
    DISPLAY_MAX = 900  # sisi terpanjang tampilan (px)

    def __init__(self, images, ai: AISuggester):
        self.images = images
        self.ai = ai
        self.idx = 0
        self.saved_for_current = []  # daftar Path crop tersimpan (untuk undo)
        self.suggestion = None
        self.box = None  # (x0,y0,x1,y1) dalam koordinat tampilan
        self._start = None
        self._rect_id = None

        common.RAW_DIR.mkdir(parents=True, exist_ok=True)
        for c in CLASSES:
            (common.RAW_DIR / c).mkdir(parents=True, exist_ok=True)

        self.root = tk.Tk()
        self.root.title("Label Assist")
        self.canvas = tk.Canvas(self.root, bg="black", cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.root.bind("<Key>", self._on_key)

        self._load()
        self.root.mainloop()

    # ---------- pemuatan & tampilan ----------
    def _load(self):
        self.saved_for_current = []
        self.box = None
        self.suggestion = None
        self._rect_id = None
        path = self.images[self.idx]
        self.pil = Image.open(path).convert("RGB")
        w, h = self.pil.size
        self.scale = min(self.DISPLAY_MAX / w, self.DISPLAY_MAX / h, 1.0)
        self.disp = self.pil.resize((int(w * self.scale), int(h * self.scale)))
        self.tk_img = ImageTk.PhotoImage(self.disp)
        self.canvas.config(width=self.disp.width, height=self.disp.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self._title()

    def _title(self):
        legend = "  ".join(f"{i+1}={c}" for i, c in enumerate(CLASSES))
        sug = f"SARAN: {self.suggestion}" if self.suggestion else "SARAN: -"
        self.root.title(
            f"[{self.idx+1}/{len(self.images)}] {self.images[self.idx].name}  |  "
            f"{sug}  |  tersimpan: {len(self.saved_for_current)}  ||  "
            f"{legend}  | Enter=saran  u=undo  n/p=navigasi  q=keluar"
        )

    # ---------- gambar kotak ----------
    def _on_press(self, e):
        self._start = (e.x, e.y)
        if self._rect_id:
            self.canvas.delete(self._rect_id)
        self._rect_id = self.canvas.create_rectangle(e.x, e.y, e.x, e.y,
                                                     outline="yellow", width=2)

    def _on_drag(self, e):
        if self._start and self._rect_id:
            self.canvas.coords(self._rect_id, self._start[0], self._start[1], e.x, e.y)

    def _on_release(self, e):
        if not self._start:
            return
        x0, y0 = self._start
        x1, y1 = e.x, e.y
        self.box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self._start = None
        if (self.box[2] - self.box[0]) < 8 or (self.box[3] - self.box[1]) < 8:
            self.box = None
            return
        crop = self._crop_original()
        cls, dbg = self.ai.suggest(crop)
        self.suggestion = f"{cls}  [{dbg}]" if cls else f"?  [{dbg}]"
        self._suggested_cls = cls
        self._title()

    def _crop_original(self) -> Image.Image:
        x0, y0, x1, y1 = self.box
        s = self.scale
        return self.pil.crop((int(x0 / s), int(y0 / s), int(x1 / s), int(y1 / s)))

    # ---------- keyboard ----------
    def _on_key(self, e):
        k = e.keysym.lower()
        if k in ("q", "escape"):
            self.root.destroy()
        elif k in ("n", "right"):
            self._next(+1)
        elif k in ("p", "left"):
            self._next(-1)
        elif k == "u":
            self._undo()
        elif k in ("return", "space"):
            if getattr(self, "_suggested_cls", None):
                self._save(self._suggested_cls)
        elif k in tuple("12345"):
            self._save(CLASSES[int(k) - 1])

    def _save(self, cls: str):
        if not self.box:
            return
        crop = self._crop_original()
        stem = self.images[self.idx].stem
        dst_dir = common.RAW_DIR / cls
        n = len(list(dst_dir.glob(f"{stem}_*.jpg")))
        dst = dst_dir / f"{stem}_{n:02d}.jpg"
        while dst.exists():
            n += 1
            dst = dst_dir / f"{stem}_{n:02d}.jpg"
        crop.save(dst, quality=92)
        self.saved_for_current.append(dst)
        # bersihkan kotak agar siap crop objek berikutnya
        if self._rect_id:
            self.canvas.delete(self._rect_id)
            self._rect_id = None
        self.box = None
        self.suggestion = None
        self._suggested_cls = None
        self._title()

    def _undo(self):
        if self.saved_for_current:
            last = self.saved_for_current.pop()
            try:
                last.unlink()
            except OSError:
                pass
            self._title()

    def _next(self, step):
        new = self.idx + step
        if 0 <= new < len(self.images):
            self.idx = new
            self._load()


def main():
    ap = argparse.ArgumentParser(description="Tool pelabelan crop + saran AI.")
    ap.add_argument("--src", default="dataset/incoming",
                    help="Folder berisi JPG hasil konversi (default: dataset/incoming).")
    ap.add_argument("--no-ai", action="store_true", help="Matikan saran AI.")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    imgs = sorted(p for p in src.rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not imgs:
        sys.exit(f"Tidak ada gambar (.jpg/.jpeg/.png) di {src}. "
                 f"Jalankan convert_heic.py dulu.")

    print(f"{len(imgs)} gambar siap dilabeli. Crop tersimpan ke dataset/raw/<kelas>/.")
    ai = AISuggester(enabled=not args.no_ai)
    Labeler(imgs, ai)
    print("Selesai. Lanjutkan dengan: python scripts/split_dataset.py")


if __name__ == "__main__":
    main()
