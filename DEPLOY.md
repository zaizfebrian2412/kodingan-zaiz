# Deploy ke VPS dengan Docker (HTTPS self-signed via IP)

Panduan menjalankan aplikasi web Klasifikasi Citra Sampah di VPS apa pun
(DigitalOcean Droplet, dll) memakai **Docker**, dengan **HTTPS self-signed**
diakses lewat **alamat IP** (tanpa domain).

> **Kenapa HTTPS?** Fitur **kamera** (`getUserMedia`) hanya aktif di *secure
> context* (HTTPS atau localhost). Dengan self-signed, kamera tetap jalan asalkan
> kamu klik **"Lanjutkan / Proceed"** sekali pada peringatan sertifikat browser.
> Lewat `http://` biasa, kamera tidak bisa (upload file tetap bisa).

---

## 1. Spesifikasi VPS minimal

| | Minimal nyaman | Mepet (butuh swap) |
|---|---|---|
| RAM | **2 GB** | 1 GB + swap 2 GB |
| vCPU | 1 | 1 |
| Disk | 25 GB | 25 GB |
| OS | Ubuntu 22.04/24.04 | sama |

> TensorFlow butuh ±0.6–1 GB RAM. **Jangan pakai 512 MB** (akan OOM saat import TF).
> Di DigitalOcean: pilih **Basic Regular $12/mo (2 GB)**. Kalau pakai 1 GB ($6),
> tambahkan swap (lihat langkah 2b).

---

## 2. Siapkan VPS

SSH ke VPS, lalu pasang Docker:

```bash
# 2a) Install Docker
curl -fsSL https://get.docker.com | sh

# 2b) (HANYA bila RAM 1 GB) tambah swap 2 GB
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Buka port HTTPS di firewall bila aktif:
```bash
sudo ufw allow 443/tcp
```

---

## 3. Ambil kode & jalankan

```bash
# Clone branch demo (yang berisi mode tanpa-training + kamera + Docker)
git clone -b demo-mobilenet-imagenet <URL-REPO-ANDA> cnn-zaiz
cd cnn-zaiz

# Build & jalankan (ganti dengan IP publik VPS Anda)
SERVER_IP=<IP-PUBLIK-VPS> docker compose up -d --build
```

Build pertama memakan beberapa menit (mengunduh TensorFlow + bobot model).

---

## 4. Akses

Buka di browser:

```
https://<IP-PUBLIK-VPS>
```

- Browser menampilkan peringatan sertifikat (karena self-signed). Klik
  **Advanced → Proceed / Lanjutkan**.
- Setelah masuk, **kamera & upload file** keduanya berfungsi.

---

## 5. Operasional

```bash
docker compose logs -f          # lihat log
docker compose restart          # restart
docker compose down             # stop & hapus container
SERVER_IP=<IP> docker compose up -d --build   # update setelah git pull
```

Data yang dipertahankan antar-restart (via volume): gambar upload dan sertifikat.

---

## Tanpa docker compose (alternatif `docker run`)

```bash
docker build -t cnn-zaiz-web .
docker run -d --name cnn-zaiz -p 443:8443 \
  -e SERVER_IP=<IP-PUBLIK-VPS> -e WASTE_DEMO=1 \
  --restart unless-stopped cnn-zaiz-web
```

---

## Catatan

- **1 worker gunicorn** sengaja dipakai karena TensorFlow boros RAM; cukup untuk
  demo/sidang (trafik rendah). Untuk konkruensi ringan sudah ada 4 thread.
- Bila nanti sudah punya **model terlatih** (`model/waste_classifier.keras`),
  hapus `WASTE_DEMO=1`, salin folder `model/` ke image (tambahkan `COPY model/`
  di Dockerfile), rebuild — aplikasi otomatis memakai model terlatih.
- Untuk production sungguhan (punya domain), ganti self-signed dengan nginx +
  Let's Encrypt (certbot) agar tanpa peringatan browser.
