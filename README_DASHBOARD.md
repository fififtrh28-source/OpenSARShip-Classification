# Dashboard Sistem Pemantauan Kapal Ikan AIS-SAR

Dashboard ini adalah prototipe sistem pemantauan kapal ikan berbasis peta untuk menampilkan hasil integrasi AIS-SAR, estimasi posisi AIS berbasis Kalman Filter, dan alert indikasi kapal ikan mencurigakan.

Konsep utama penelitian:

- Kalman Filter hanya digunakan untuk mengoreksi atau menghaluskan posisi AIS.
- Data SAR tetap diambil dari metadata/deteksi SAR, misalnya `SAR_Latitude`, `SAR_Longitude`, atau fallback `Center_latitude`, `Center_longitude`.
- Fusion/matching dilakukan dengan membandingkan posisi AIS hasil Kalman (`Kalman_Lat`, `Kalman_Lon`) terhadap posisi SAR.
- Data AIS yang tidak memiliki estimasi Kalman tidak dipakai untuk matching/fusion, tetapi diberi status verifikasi seperti `AIS tanpa estimasi Kalman` atau `Data AIS tidak lengkap untuk Kalman`.
- Dashboard final secara default hanya menampilkan kapal dengan kategori `Fishing`, `Fishing Vessel`, `Fish`, atau `Kapal Ikan`.

Alert pada dashboard ini bukan vonis IUU Fishing dan bukan sistem distress/SSAS kapal. Alert digunakan sebagai peringatan dini untuk investigasi lanjutan.

## 1. Instalasi

Jalankan dari folder repository:

```bash
pip install -r requirements_dashboard.txt
```

Jika menggunakan virtual environment lokal repository di Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_dashboard.txt
```

## 2. Menjalankan Dashboard

```bash
streamlit run app_dashboard.py
```

Atau dengan virtual environment lokal Windows:

```powershell
.\.venv\Scripts\streamlit.exe run app_dashboard.py
```

## 3. File Input

Dashboard membaca file yang tersedia secara adaptif. File yang diprioritaskan:

- `dataset/fusion_kalman_with_type.csv`
- `dataset/fusion_minimal.csv`
- `fusion_minimal.csv`
- `dataset/fusion_minimal_with_type.csv`
- `dataset/metadata.csv`
- `kalman_ais_result.csv`

Kode dashboard tidak mengarang nama kolom. Fungsi normalisasi kolom akan mencoba menyesuaikan alias seperti:

- ship type: `Ship_Type`, `Ship_type`, `ship_type`, `category`, `vessel_type`, `Vessel_Type`
- AIS: `AIS_Latitude`, `AIS_Longitude`, `Sog`, `Cog`
- SAR: `SAR_Latitude`, `SAR_Longitude`, `Center_latitude`, `Center_longitude`
- Kalman: `Kalman_Lat`, `Kalman_Lon`, `Kalman_Shift_m`
- prediksi SAR jika ada: `predicted_class`, `sar_class`, `sar_ship_type`, `classification`

## 4. Filter Kapal Ikan

Dashboard membuat kolom:

- `ship_type_normalized`
- `is_fishing_vessel`
- `is_sar_predicted_fishing`

`is_fishing_vessel = True` jika nilai ship type mengandung:

- `fishing`
- `fish`
- `kapal ikan`

Checkbox sidebar `Tampilkan hanya kapal ikan` aktif secara default. Jika aktif, semua summary cards, marker peta, fusion, dan alert dihitung hanya dari kapal ikan.

## 5. File Output

Saat dashboard dijalankan, sistem membuat folder `output/` dan menyimpan:

- `output/dashboard_kapal_ikan_ais_sar.csv`
- `output/alert_kapal_ikan_ais_sar.csv`

Untuk membuat output CSV tanpa membuka dashboard Streamlit:

```powershell
.\.venv\Scripts\python.exe -B app_dashboard.py --build-output
```

## 6. Kategori Marker

Dashboard menggunakan layer peta berikut:

- **AIS Kapal Ikan Mentah**: marker biru muda dari `AIS_Latitude`, `AIS_Longitude`.
- **AIS Kapal Ikan Hasil Kalman**: marker cyan/ungu dari `Kalman_Lat`, `Kalman_Lon`.
- **SAR Kapal Ikan**: marker oranye dari `SAR_Latitude`, `SAR_Longitude` pada baris kapal ikan/fusion kapal ikan.
- **Fusion Kapal Ikan**: marker hijau untuk matched AIS fishing + SAR berdasarkan threshold jarak.
- **Alert Kapal Ikan**: marker merah untuk kapal ikan yang perlu investigasi.

## 7. Logika SAR dan Alert

SAR dianggap terkait kapal ikan jika baris data memiliki kategori kapal ikan atau ada kolom prediksi SAR seperti `predicted_class`, `sar_class`, `sar_ship_type`, atau `classification` yang bernilai fishing/fishing vessel.

SAR tanpa AIS tetapi tidak memiliki label/prediksi fishing tidak langsung disebut sebagai kandidat dark vessel kapal ikan. Status yang digunakan adalah:

```text
SAR tanpa AIS - tipe kapal belum terklasifikasi
```

Alert kapal ikan dibuat jika:

1. Kapal ikan AIS hasil Kalman memiliki jarak ke SAR lebih besar dari threshold sidebar.
2. Kapal ikan memiliki `Kalman_Shift_m` lebih besar dari threshold sidebar.
3. SOG kapal ikan tidak normal, misalnya melebihi threshold sidebar.
4. SAR terdeteksi sebagai fishing vessel tetapi tidak punya pasangan AIS.

Nama alert yang digunakan:

- `Anomali Posisi Kapal Ikan AIS-SAR`
- `Kapal Ikan Perlu Investigasi`
- `Kandidat Dark Vessel Kapal Ikan`
- `Indikasi Aktivitas Mencurigakan`

Dashboard tidak menulis “pasti IUU Fishing”.

## 8. Penjelasan Akademik

Dashboard ini hanya menampilkan kapal dengan kategori Fishing/Fishing Vessel. Alert yang ditampilkan merupakan indikasi awal untuk investigasi, bukan vonis IUU Fishing.

Dalam konteks Maritime Situational Awareness, dashboard ini digunakan untuk membantu melihat hubungan antara AIS kapal ikan, deteksi SAR, estimasi posisi AIS berbasis Kalman Filter, dan indikator awal kapal ikan yang perlu investigasi lanjutan.
