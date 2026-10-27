import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

# Konfigurasi Halaman
st.set_page_config(
    page_title="Sistem Pendukung Keputusan - Prediksi Risiko Diabetes",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk mempercantik tampilan agar lebih rapi dan profesional
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .card-container {
        background-color: #F8FAFC;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
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

  # SECTION 1: INPUT DATA PASIEN BARU (Full Width / Lebar Penuh)
  st.subheader("📝 Input Data Pasien Baru")
  with st.container():
    with st.form("form_prediksi"):
      c1, c2 = st.columns(2, gap="large")
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

      st.markdown("")
      submitted = st.form_submit_button(
          "🔍 Analisis Risiko Pasien", type="primary", use_container_width=True
      )

  st.markdown("<br>", unsafe_allow_html=True)

  # SECTION 2: HASIL EVALUASI MODEL (Tepat di bawah input, lebar penuh & leluasa)
  st.subheader("📊 Hasil Evaluasi Model")

  if submitted:
    data_baru = np.array([[usia, bmi, glukosa, td, riwayat]])
    prediksi = knn.predict(data_baru)[0]
    probabilitas = knn.predict_proba(data_baru)[0]
    jarak, indeks_tetangga = knn.kneighbors(data_baru)

    # Menampilkan hasil dalam container agar tampak rapi
    with st.container():
      if prediksi == 1:
        st.error(
            "### Kesimpulan: RISIKO TINGGI (Kelas 1)\nPasien diindikasikan"
            " memiliki karakteristik yang masuk dalam kelompok risiko tinggi"
            " berdasarkan parameter klinis yang dimasukkan."
        )
      else:
        st.success(
            "### Kesimpulan: RISIKO RENDAH (Kelas 0)\nPasien diindikasikan"
            " memiliki karakteristik yang masuk dalam kelompok risiko rendah."
        )

      m1, m2 = st.columns(2)
      with m1:
        st.metric(label="Parameter K-NN", value=f"K = {k_val}")
      with m2:
        st.metric(
            label="Tingkat Keyakinan Model",
            value=f"{max(probabilitas)*100:.1f}%",
        )

      st.markdown("---")
      with st.expander(
          "🔎 Lihat Detail Tetangga Terdekat (Nearest Neighbors)", expanded=True
      ):
        st.write(
            "Data historis pasien terdekat yang menjadi acuan perhitungan"
            " kemiripan jarak (Euclidean Distance):"
        )
        pasien_terdekat = df_latih.iloc[indeks_tetangga[0]]
        st.dataframe(pasien_terdekat, use_container_width=True)
  else:
    st.info(
        "👆 Silakan isi formulir parameter klinis di atas, lalu klik tombol"
        " **Analisis Risiko Pasien** untuk melihat hasil evaluasi secara"
        " leluasa di sini."
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
