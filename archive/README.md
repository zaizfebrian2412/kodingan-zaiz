# Arsip Script Pipeline

Folder ini menyimpan script lama untuk menyiapkan dataset dan melatih model.
Script-script ini dipindah ke sini supaya folder utama lebih rapi.

Mulai sekarang, proses lengkap (siapkan file, training, sampai jalankan model)
dilakukan lewat notebook:

    notebooks/training_model.ipynb

Isi folder ini hanya untuk referensi:

- convert_heic.py   : ubah foto .HEIC iPhone jadi .jpg
- resize_images.py  : perkecil ukuran file gambar
- label_assist.py   : bantu memotong dan melabeli objek per foto
- dataset_stats.py  : hitung jumlah gambar per kelas
- split_dataset.py  : bagi dataset jadi train, validation, test
- train_model.py    : latih model dari terminal
- evaluate_model.py : evaluasi model
- predict_image.py  : prediksi satu gambar dari terminal

Catatan: sebagian script di sini meng-import scripts/common.py. Jika ingin
menjalankannya lagi, pastikan file scripts/common.py masih ada.
