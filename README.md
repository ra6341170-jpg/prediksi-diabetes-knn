# 🩺 Sistem Pendukung Keputusan (SPK) - Prediksi Risiko Diabetes dengan KNN

Aplikasi web interaktif berbasis Python dan [Streamlit](https://streamlit.io/) untuk mengidentifikasi tingkat risiko diabetes pada pasien menggunakan algoritma *K-Nearest Neighbors (KNN)*. Proyek ini dikembangkan sebagai bagian dari penugasan akademik mata kuliah Data Mining.

🌐 **Akses Aplikasi Secara Online:** 👉 [Link Web Streamlit Prediksi Risiko Diabetes](https://prediksi-diabetes-knn-odkxzsgx7bq36gtfnzumrf.streamlit.app/)[cite: 1]

---

## 📋 Fitur Utama
- **Multi-Page Layout & Desain Modern**: Antarmuka terstruktur dengan sidebar navigasi interaktif ala aplikasi enterprise.
- **Dynamic Data Loading:** Memuat dataset historis pasien secara otomatis langsung dari file eksternal (`data-pasien.xlsx`).
- **Parameter K Dinamis:** Pengguna dapat mengatur nilai $K$ (jumlah tetangga terdekat) secara fleksibel melalui *slider* interaktif (1 s.d. 9).
- **Ilustrasi Grafik Sebaran & Pemilihan Fitur Sumbu:** Fitur visualisasi sebaran data pelatihan dengan pilihan atribut sumbu X dan Y yang dapat diatur bebas oleh pengguna.
- **Transparansi Evaluasi & Tabel Jarak:** Menampilkan tingkat keyakinan model serta tabel rincian tetangga terdekat lengkap dengan hasil perhitungan **Jarak Euclidean** yang diletakkan pada kolom paling akhir.
- **Eksplorasi Dataset:** Halaman khusus untuk melihat tabel data latih historis beserta ringkasan statistik deskriptifnya.
- **Penjelasan Metode yang Ramah Pengguna:** Dilengkapi analogi sederhana agar siapa pun dapat memahami cara kerja algoritma KNN dengan mudah tanpa harus pusing melihat rumus rumit.

---

## 🛠️ Atribut Dataset & Parameter Klinis
Sistem ini memproses 5 parameter klinis utama dari data pasien:
1. **Usia** (Tahun)
2. **BMI** (Indeks Massa Tubuh)
3. **Glukosa** (Kadar gula darah)
4. **Tekanan Darah** (mmHg)
5. **Riwayat Keluarga** (0 = Tidak Ada, 1 = Ada)

**Target Klasifikasi:**
- `0` = Risiko Rendah
- `1` = Risiko Tinggi

---

## 📐 Algoritma & Perhitungan (Euclidean Distance)
Untuk mengukur tingkat kemiripan antara data pasien baru dengan data historis pada dataset, sistem menggunakan rumus jarak Euclidean:

$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

---

## 📂 Struktur Folder Proyek
prediksi-diabetes-knn/
│
├── .gitignore          # Konfigurasi file/folder yang diabaikan Git
├── requirements.txt    # Daftar dependensi library Python (streamlit, scikit-learn, pandas, matplotlib, openpyxl)
├── README.md           # Dokumentasi proyek
├── data-pasien.xlsx    # Dataset acuan pasien (Excel)
└── app.py              # Kode utama aplikasi web Streamlit
