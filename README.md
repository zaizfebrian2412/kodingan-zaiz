# Sistem Klasifikasi Citra Sampah Lokal Berbasis CNN

Prototype skripsi untuk **klasifikasi citra sampah** menggunakan **Convolutional
Neural Network (CNN)** dengan arsitektur **MobileNetV2 (transfer learning)** dan
penerapan **Data Augmentation**.

> **Studi Kasus:** UPT TPSA Cihara.

Aplikasi web sederhana (Flask) memungkinkan pengguna mengunggah foto sampah, lalu
sistem menampilkan **jenis sampah**, **tingkat keyakinan (confidence)**, dan
**probabilitas semua kelas**.

---

## 1. Deskripsi Project

Alur utama aplikasi:

1. Pengguna membuka halaman web.
2. Pengguna mengunggah foto sampah.
3. Sistem melakukan preprocessing gambar (resize 224×224, RGB, normalisasi MobileNetV2).
4. Sistem menjalankan model CNN yang sudah dilatih (hanya inference, tanpa training ulang).
5. Sistem menampilkan label, confidence score, dan preview gambar.

Kategori (kelas) awal: **organik, plastik, kertas, logam, kaca**.
Jumlah kelas **fleksibel** — sistem otomatis membaca folder yang ada di `dataset/raw/`.
Jika menambah kelas baru (mis. `b3`, `karet`, `lainnya`), cukup tambahkan foldernya.

---

## 2. Struktur Folder

```
cnn-zaiz/
├── app/
│   ├── app.py                  # Aplikasi Flask (upload & prediksi)
│   ├── static/
│   │   ├── uploads/            # Gambar yang diunggah pengguna
│   │   └── css/style.css       # Tampilan (tema hijau)
│   └── templates/
│       ├── index.html          # Halaman upload
│       └── result.html         # Halaman hasil
├── dataset/
│   ├── raw/                    # Dataset mentah per kelas (Anda isi sendiri)
│   │   ├── organik/  plastik/  kertas/  logam/  kaca/
│   ├── train/                  # Hasil split (otomatis)
│   ├── validation/             # Hasil split (otomatis)
│   └── test/                   # Hasil split (otomatis)
├── model/
│   ├── waste_classifier.keras  # Model terlatih (dihasilkan training)
│   └── labels.json             # Pemetaan indeks -> nama kelas
├── notebooks/
│   └── training_model.ipynb    # Notebook untuk Google Colab
├── scripts/
│   ├── common.py               # Fungsi bersama (preprocessing, model, dll)
│   ├── split_dataset.py        # Bagi dataset train/validation/test
│   ├── train_model.py          # Latih model
│   ├── evaluate_model.py       # Evaluasi model
│   └── predict_image.py        # Prediksi satu gambar
├── outputs/
│   ├── training_accuracy.png
│   ├── training_loss.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 3. Instalasi Dependency

> **Catatan Python:** Disarankan **Python 3.10 atau 3.11**. Python 3.9 cukup tua untuk
> versi TensorFlow terbaru. Pada macOS Apple Silicon, gunakan venv Python 3.10/3.11.

Pilih **salah satu** cara berikut.

### Opsi A — Conda (disarankan, sudah teruji di project ini)

```bash
# Buat environment Python 3.11 lalu aktifkan
conda create -y -n cnn-zaiz python=3.11
conda activate cnn-zaiz

# Install dependency
pip install --upgrade pip
pip install -r requirements.txt
```

Setiap kali membuka terminal baru, aktifkan dulu environment:
`conda activate cnn-zaiz`

### Opsi B — venv (Python bawaan)

```bash
# Pastikan menggunakan Python 3.10/3.11
python3.11 -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows

# Upgrade pip lalu install dependency
pip install --upgrade pip
pip install -r requirements.txt
```

> **Opsional (Mac Apple Silicon, akselerasi GPU Metal):**
> `pip install tensorflow-metal`

> **Catatan kompatibilitas:** Project ini sudah diuji pada **Python 3.11 +
> TensorFlow 2.16.2 + Keras 3.14** (macOS Apple Silicon). Hindari Python 3.9
> karena terlalu tua untuk versi TensorFlow terbaru.

---

## 4. Menyiapkan Dataset

Letakkan gambar pada folder per kelas di dalam `dataset/raw/`:

```
dataset/raw/organik/   foto1.jpg, foto2.jpg, ...
dataset/raw/plastik/   ...
dataset/raw/kertas/    ...
dataset/raw/logam/     ...
dataset/raw/kaca/      ...
```

Saran: minimal puluhan–ratusan gambar per kelas agar hasil lebih baik. Format yang
didukung: `.jpg`, `.jpeg`, `.png`.

### 4a. Foto masih `.HEIC` (iPhone)?

Pipeline hanya menerima `.jpg/.jpeg/.png`. Konversi dulu (memakai `sips` bawaan
macOS, tanpa install tambahan):

```bash
# unduh dulu folder foto dari Google Drive ke Mac, lalu:
python scripts/convert_heic.py --src ~/Downloads/foto_sampah --dst dataset/incoming
```

Hasil `.jpg` masuk ke `dataset/incoming/`.

### 4b. Foto belum dilabeli & berisi banyak objek (mixed)?

Foto TPSA biasanya berisi **banyak jenis sampah dalam satu frame**. Untuk
klasifikasi (1 gambar = 1 kelas) hasil terbaik diperoleh dengan **memotong tiap
objek** dan melabelinya. Gunakan tool pelabelan semi-otomatis:

```bash
python scripts/label_assist.py --src dataset/incoming
```

- Drag kotak mengelilingi **satu** objek sampah pada foto.
- AI (MobileNetV2/ImageNet) memberi **saran** kelas di judul jendela.
- Tekan `1`–`5` untuk menyimpan crop ke `dataset/raw/<kelas>/` (atau `Enter` =
  terima saran AI). Ulangi untuk objek lain pada foto yang sama.
- `n`/`p` ganti foto, `u` undo crop terakhir, `q` keluar. Tambah `--no-ai` untuk
  mematikan saran AI.

> Saran AI hanya bantuan awal dan **bisa salah** — Anda tetap memverifikasi setiap
> label. Setelah folder `dataset/raw/<kelas>/` terisi, lanjut ke split (section 5).

> **Folder kerja:** taruh semua JPG mentah (hasil konversi) di `dataset/incoming/`.
> Folder ini sudah dibuat dan diabaikan git, jadi aman sebagai tempat "dump".

### 4c. Cek jumlah gambar per kelas

Sebelum training, pastikan tiap kelas punya cukup gambar dan relatif seimbang:

```bash
python scripts/dataset_stats.py
```

Menampilkan jumlah per kelas + grafik batang sederhana dan penilaian
(`KOSONG` / `kurang <50` / `cukup ≥50` / `baik ≥150`), serta peringatan bila ada
kelas kosong atau dataset timpang (selisih antar kelas > 2×). Target praktis:
**minimal ~50, idealnya ≥150 gambar per kelas**, dengan jumlah yang mirip.

---

## 5. Split Dataset (train / validation / test)

```bash
python scripts/split_dataset.py
```

Rasio default 70% train, 15% validation, 15% test (dibagi per kelas).

Opsi tambahan:

```bash
python scripts/split_dataset.py --ratios 0.8 0.1 0.1 --seed 7
python scripts/split_dataset.py --clean       # bersihkan hasil split lama dulu
python scripts/split_dataset.py --move        # pindahkan file (bukan menyalin)
```

---

## 6. Training Model

```bash
python scripts/train_model.py
```

Parameter dapat diubah:

```bash
python scripts/train_model.py --epochs 30 --batch-size 16 --lr 0.0001 --img-size 224
```

Hasil:
- Model terbaik tersimpan di `model/waste_classifier.keras`
- Label kelas tersimpan di `model/labels.json`
- Grafik di `outputs/training_accuracy.png` dan `outputs/training_loss.png`

---

## 7. Evaluasi Model

```bash
python scripts/evaluate_model.py
```

Menghasilkan **accuracy, precision, recall, F1-score**, dan **confusion matrix**:
- `outputs/classification_report.txt`
- `outputs/confusion_matrix.png`

---

## 8. Cara Membuka / Menjalankan Program (Aplikasi Web)

Setelah model tersedia di `model/waste_classifier.keras` (hasil training di section 6
atau hasil unduhan dari Colab di section 10), jalankan aplikasi web seperti berikut:

```bash
# 1) Aktifkan environment (sekali tiap buka terminal baru)
conda activate cnn-zaiz
# atau: source venv/bin/activate

# 2) Jalankan server Flask
python app/app.py
```

Terminal akan menampilkan baris seperti:

```
 * Running on http://127.0.0.1:5000
```

**Buka alamat tersebut di browser:** **http://127.0.0.1:5000**

Cara pakai di halaman web:
1. Klik **Pilih File**, lalu pilih foto sampah (`.jpg`, `.jpeg`, atau `.png`).
2. Klik tombol **Klasifikasikan**.
3. Sistem menampilkan **jenis sampah**, **confidence**, dan **probabilitas semua kelas**.

Untuk **menghentikan** server, tekan `Ctrl + C` di terminal.

### 8a. Mode Demo (tanpa training) — MobileNetV2/ImageNet

> Hanya tersedia di branch **`demo-mobilenet-imagenet`**.

Agar UI bisa langsung dicoba **tanpa melatih model**, aplikasi punya **Mode Demo**
yang memakai **MobileNetV2 + bobot ImageNet** sebagai basis klasifikasi. Probabilitas
1000 kelas ImageNet dipetakan ke 5 kelas sampah (mis. *water_bottle* → plastik,
*carton* → kertas, *tin can* → logam, *beer_bottle* → kaca, buah/sayur → organik).

- **Otomatis aktif** bila `model/waste_classifier.keras` belum ada — cukup jalankan
  `python app/app.py`, lalu buka http://127.0.0.1:5000.
- **Dipaksa aktif** (walau model terlatih ada) dengan variabel lingkungan:
  ```bash
  WASTE_DEMO=1 python app/app.py
  ```
- Halaman menampilkan label **Mode Demo**. Hasilnya **perkiraan** (heuristik ImageNet),
  bukan model final skripsi. Untuk hasil resmi tetap latih model sendiri (section 6).

Logika ada di `scripts/demo_classifier.py`. Bila model terlatih tersedia dan
`WASTE_DEMO` tidak diset, aplikasi otomatis memakai model terlatih tersebut.

---

## 9. Prediksi Satu Gambar (CLI)

```bash
python scripts/predict_image.py --image sample.jpg
```

Contoh output:

```
Prediksi  : plastik
Confidence: 92.45%

Probabilitas:
  plastik     : 92.45%
  kertas      : 3.10%
  logam       : 2.00%
  kaca        : 1.25%
  organik     : 1.20%
```

---

## 10. Menjalankan di Google Colab

Buka `notebooks/training_model.ipynb` di [Google Colab](https://colab.research.google.com/).

Langkah ringkas:
1. Aktifkan GPU: **Runtime → Change runtime type → GPU**.
2. Unggah dataset (mount Google Drive atau upload `dataset.zip`) — lihat sel "Load Dataset".
3. Jalankan sel berurutan hingga selesai.
4. Unduh `waste_classifier.keras` dan `labels.json`, lalu letakkan keduanya di folder
   `model/` pada project lokal agar bisa dipakai aplikasi Flask & script prediksi.

---

## 11. Urutan Perintah Lengkap (Ringkasan)

```bash
# 1) Siapkan environment (pilih conda ATAU venv)
conda create -y -n cnn-zaiz python=3.11 && conda activate cnn-zaiz
# atau: python3.11 -m venv venv && source venv/bin/activate

pip install -r requirements.txt

# 2) Isi dataset/raw/<kelas>/ dengan gambar, lalu:
python scripts/split_dataset.py
python scripts/train_model.py
python scripts/evaluate_model.py
python app/app.py
```

---

## 12. Troubleshooting Umum

| Masalah | Penyebab & Solusi |
|---|---|
| `ModuleNotFoundError: No module named 'tensorflow'` | Environment belum aktif (`conda activate cnn-zaiz`) atau dependency belum terinstall (`pip install -r requirements.txt`). |
| `Folder dataset kosong / tidak ada` | Belum mengisi `dataset/raw/` atau belum menjalankan `split_dataset.py`. |
| `Model belum tersedia` di Flask | Belum melakukan training. Jalankan `python scripts/train_model.py` (atau salin model dari Colab ke `model/`). |
| TensorFlow gagal install di Python 3.9 | Gunakan Python 3.10/3.11 pada virtual environment baru. |
| Error numpy / versi tidak cocok | Pastikan `numpy<2.0` (sudah diatur di `requirements.txt`). |
| Training sangat lambat | Wajar di CPU. Gunakan GPU di Google Colab untuk training lebih cepat. |
| Akurasi rendah | Tambah jumlah & variasi gambar per kelas, atau naikkan jumlah epoch. |

---

## Catatan Teknis

- **Preprocessing dibakar ke dalam model:** layer `Rescaling(1/127.5, offset=-1)`
  (setara `mobilenet_v2.preprocess_input`) berada di dalam model, sehingga proses
  prediksi cukup memberi piksel mentah RGB `(1, 224, 224, 3)` — konsisten dan
  bebas bug *double preprocessing*.
- **Data augmentation** hanya aktif saat training (otomatis non-aktif saat inference).
- **Label tidak di-hardcode:** dibaca otomatis dari folder dataset dan disimpan ke
  `model/labels.json`.

> Prototype ini ditujukan untuk kebutuhan skripsi, bukan sistem produksi skala besar.
