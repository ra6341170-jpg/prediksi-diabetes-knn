# 🩺 Sistem Pendukung Keputusan (SPK) - Prediksi Risiko Diabetes dengan KNN

Aplikasi web interaktif berbasis Python dan Streamlit untuk mengidentifikasi tingkat risiko diabetes pada pasien menggunakan algoritma **K-Nearest Neighbors (KNN)**. Proyek ini dikembangkan sebagai bagian dari penugasan akademik mata kuliah Data Mining.

---

## 📋 Fitur Utama
- **Multi-Page Layout & Desain Modern**: Antarmuka terstruktur dengan sidebar navigasi interaktif ala aplikasi enterprise.
- **Dynamic Data Loading**: Memuat dataset historis secara otomatis langsung dari file eksternal (`data-pasien.xlsx`).
- **Algoritma K-Nearest Neighbors (KNN)**: Menggunakan $K = 3$ untuk mengklasifikasikan risiko pasien berdasarkan kemiripan jarak fitur klinis.
- **Transparansi Evaluasi**: Menampilkan tingkat keyakinan (*confidence level*) model serta rincian tetangga terdekat (*Nearest Neighbors*) secara transparan.

---

## 🛠️ Atribut Dataset & Parameter
Sistem ini memproses 5 parameter klinis utama dari data pasien:
1. **Usia** (Tahun)
2. **BMI** (Indeks Massa Tubuh)
3. **Glukosa** (Kadar gula darah)
4. **Tekanan Darah** (mmHg)
5. **Riwayat Keluarga** (`0` = Tidak Ada, `1` = Ada)

**Target Klasifikasi:**
- `0` = Risiko Rendah
- `1` = Risiko Tinggi

---

## 📂 Struktur Folder Proyek
```text
prediksi-diabetes-knn/
│
├── .gitignore          # Konfigurasi file/folder yang diabaikan Git
├── requirements.txt    # Daftar dependensi library Python
├── README.md           # Dokumentasi proyek
├── data-pasien.xlsx    # Dataset acuan pasien (Excel)
└── app.py              # Kode utama aplikasi web Streamlit