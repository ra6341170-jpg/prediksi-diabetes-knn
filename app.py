import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

# Konfigurasi Halaman
st.set_page_config(
    page_title="Sistem Pendukung Keputusan - Prediksi Risiko Diabetes",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk mempercantik tampilan
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Sidebar Navigasi & Informasi
with st.sidebar:
  st.image("https://img.icons8.com/color/96/diabetes.png", width=80)
  st.title("Menu Utama")
  menu = st.radio(
      "Pilih Halaman:",
      ["Dashboard & Prediksi", "Eksplorasi Dataset", "Tentang Metode KNN"],
  )
  st.markdown("---")
  st.markdown("**Dibuat oleh:** Rizky Ardiansyah")
  st.markdown("**Mata Kuliah:** Data Mining")


# Memuat Dataset dari file Excel secara otomatis
@st.cache_data
def load_data():
  # Membaca file excel 'data-pasien.xlsx'
  df = pd.read_excel("data-pasien.xlsx", sheet_name="Data Pasien")
  return df


try:
  df_latih = load_data()
except Exception as e:
  st.error(
      f"Gagal memuat file 'data-pasien.xlsx'. Pastikan file berada di folder"
      f" yang sama! Error: {e}"
  )
  st.stop()

# Mempersiapkan fitur (X) dan target (y) untuk KNN
X = df_latih[["Usia", "BMI", "Glukosa", "Tekanan_Darah", "Riwayat_Keluarga"]]
y = df_latih["Kelas"]

# Inisialisasi Model KNN dengan K=3
k_val = 3
knn = KNeighborsClassifier(n_neighbors=k_val)
knn.fit(X, y)


# --- HALAMAN 1: DASHBOARD & PREDIKSI ---
if menu == "Dashboard & Prediksi":
  st.markdown(
      '<p class="main-header">Sistem Prediksi Risiko Diabetes</p>',
      unsafe_allow_html=True,
  )
  st.markdown(
      '<p class="sub-header">Implementasi Algoritma K-Nearest Neighbors (KNN)'
      " Berbasis Dataset Klinis Pasien.</p>",
      unsafe_allow_html=True,
  )
  st.markdown("---")

  col_input, col_result = st.columns([1.2, 1], gap="large")

  with col_input:
    st.subheader("📝 Input Data Pasien Baru")
    with st.form("form_prediksi"):
      c1, c2 = st.columns(2)
      with c1:
        usia = st.number_input("Usia (Tahun)", min_value=1, max_value=100, value=42)
        bmi = st.number_input(
            "Indeks Massa Tubuh (BMI)",
            min_value=10.0,
            max_value=60.0,
            value=28.5,
        )
        glukosa = st.number_input(
            "Kadar Glukosa", min_value=50, max_value=300, value=130
        )
      with c2:
        td = st.number_input(
            "Tekanan Darah (mmHg)", min_value=50, max_value=250, value=128
        )
        riwayat = st.selectbox(
            "Riwayat Diabetes Keluarga",
            options=[0, 1],
            format_func=lambda x: (
                "Ada (1)" if x == 1 else "Tidak Ada (0)"
            ),
        )

      submitted = st.form_submit_button(
          "🔍 Analisis Risiko Pasien", type="primary", use_container_width=True
      )

  with col_result:
    st.subheader("📊 Hasil Evaluasi Model")
    if submitted:
      data_baru = np.array([[usia, bmi, glukosa, td, riwayat]])
      prediksi = knn.predict(data_baru)[0]
      probabilitas = knn.predict_proba(data_baru)[0]
      jarak, indeks_tetangga = knn.kneighbors(data_baru)

      if prediksi == 1:
        st.error("### Kesimpulan: RISIKO TINGGI")
        st.markdown(
            "Pasien diindikasikan memiliki karakteristik yang masuk dalam"
            " kelompok **Risiko Tinggi** (Kelas 1)."
        )
      else:
        st.success("### Kesimpulan: RISIKO RENDAH")
        st.markdown(
            "Pasien diindikasikan memiliki karakteristik yang masuk dalam"
            " kelompok **Risiko Rendah** (Kelas 0)."
        )

      st.markdown("---")
      st.markdown(f"**Parameter K-NN:** K = {k_val}")
      st.markdown(
          f"**Tingkat Keyakinan Model:**"
          f" {max(probabilitas)*100:.1f}%"
      )

      # Menampilkan tetangga terdekat
      with st.expander("🔎 Lihat Detail Tetangga Terdekat (Nearest Neighbors)"):
        pasien_terdekat = df_latih.iloc[indeks_tetangga[0]]
        st.dataframe(pasien_terdekat, use_container_width=True)
    else:
      st.info(
          "Silakan masukkan parameter klinis pasien di sebelah kiri dan klik"
          " tombol **Analisis Risiko Pasien** untuk melihat hasil prediksi."
      )


# --- HALAMAN 2: EKSPLORASI DATASET ---
elif menu == "Eksplorasi Dataset":
  st.markdown(
      '<p class="main-header">Eksplorasi Dataset Latih</p>',
      unsafe_allow_html=True,
  )
  st.write(
      "Berikut adalah data historis pasien yang dimuat langsung dari file"
      " **data-pasien.xlsx**:"
  )

  st.dataframe(df_latih, use_container_width=True)

  st.markdown("### Ringkasan Statistik Data")
  st.dataframe(df_latih.describe(), use_container_width=True)


# --- HALAMAN 3: TENTANG METODE KNN ---
elif menu == "Tentang Metode KNN":
  st.markdown(
      '<p class="main-header">Tentang Metode K-Nearest Neighbors (KNN)</p>',
      unsafe_allow_html=True,
  )
  st.write(
      "**K-Nearest Neighbors (KNN)** adalah algoritma supervised learning yang"
      " bekerja dengan mencari jarak terdekat antara data uji (pasien baru)"
      " dengan sejumlah $k$ data pelatihan terdekat."
  )

  st.markdown("#### Langkah Kerja Sistem:")
  st.markdown(
      "1. **Menentukan Nilai K:** Pada sistem ini digunakan nilai $K = 3$."
  )
  st.markdown(
      "2. **Menghitung Jarak:** Mengukur jarak kemiripan menggunakan jarak"
      " Euclidean antara pasien baru dengan seluruh data historis di file"
      " Excel."
  )
  st.markdown(
      "3. **Mayoritas Voting:** Mengambil 3 data historis yang memiliki jarak"
      " terdekat, lalu menentukan kelas mayoritas sebagai hasil akhir"
      " prediksi."
  )
