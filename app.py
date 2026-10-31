import matplotlib.pyplot as plt
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
st.markdown("""
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
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigasi & Informasi
with st.sidebar:
  st.image("https://img.icons8.com/color/96/diabetes.png", width=80)
  st.title("Menu Utama")
  menu = st.radio(
      "Pilih Halaman:",
      [
          "Dashboard & Prediksi",
          "Ilustrasi Grafik KNN",
          "Eksplorasi Dataset",
          "Tentang Metode KNN",
      ],
  )
  st.markdown("---")
  st.markdown("**Dibuat oleh:** Rizky Ardiansyah")
  st.markdown("**Mata Kuliah:** Data Mining / SPK")


# Memuat Dataset dari file Excel secara otomatis
@st.cache_data
def load_data():
  df = pd.read_excel("data-pasien.xlsx", sheet_name="Data Pasien")
  return df


try:
  df_latih = load_data()
except Exception as e:
  st.error(f"Gagal memuat file 'data-pasien.xlsx'. Pastikan file berada di folder yang sama! Error: {e}")
  st.stop()

# Mempersiapkan fitur (X) dan target (y) untuk KNN
X = df_latih[["Usia", "BMI", "Glukosa", "Tekanan_Darah", "Riwayat_Keluarga"]]
y = df_latih["Kelas"]


# --- HALAMAN 1: DASHBOARD & PREDIKSI ---
if menu == "Dashboard & Prediksi":
  st.markdown('<p class="main-header">Sistem Prediksi Risiko Diabetes</p>', unsafe_allow_html=True)
  st.markdown('<p class="sub-header">Implementasi Algoritma K-Nearest Neighbors (KNN) dengan Parameter K Dinamis.</p>', unsafe_allow_html=True)
  st.markdown("---")

  st.subheader("📝 Input Data Pasien Baru & Parameter KNN")
  with st.container():
    with st.form("form_prediksi"):
      c1, c2 = st.columns(2, gap="large")
      with c1:
        usia = st.number_input("Usia (Tahun)", min_value=1, max_value=100, value=42)
        bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=10.0, max_value=60.0, value=28.5)
        glukosa = st.number_input("Kadar Glukosa", min_value=50, max_value=300, value=130)
      with c2:
        td = st.number_input("Tekanan Darah (mmHg)", min_value=50, max_value=250, value=128)
        riwayat = st.selectbox(
            "Riwayat Diabetes Keluarga",
            options=[0, 1],
            format_func=lambda x: "Ada (1)" if x == 1 else "Tidak Ada (0)"
        )
        
        k_val = st.slider(
            "Pilih Nilai K (Jumlah Tetangga Terdekat)", 
            min_value=1, 
            max_value=9, 
            step=2, 
            value=3,
            help="Disarankan memilih angka ganjil (1, 3, 5, dst.) untuk menghindari kesamaan suara (tie)."
        )

      st.markdown("")
      submitted = st.form_submit_button("🔍 Analisis Risiko Pasien", type="primary", use_container_width=True)

  st.markdown("<br>", unsafe_allow_html=True)

  st.subheader("📊 Hasil Evaluasi Model")

  if submitted:
    if k_val > len(df_latih):
      st.error(f"Nilai K ({k_val}) tidak boleh melebihi jumlah total data latih ({len(df_latih)} data)!")
    else:
      knn = KNeighborsClassifier(n_neighbors=k_val)
      knn.fit(X, y)

      data_baru = np.array([[usia, bmi, glukosa, td, riwayat]])
      prediksi = knn.predict(data_baru)[0]
      probabilitas = knn.predict_proba(data_baru)[0]
      jarak, indeks_tetangga = knn.kneighbors(data_baru)

      with st.container():
        if prediksi == 1:
          st.error("### Kesimpulan: RISIKO TINGGI (Kelas 1)\nPasien diindikasikan memiliki karakteristik yang masuk dalam kelompok risiko tinggi berdasarkan parameter klinis.")
        else:
          st.success("### Kesimpulan: RISIKO RENDAH (Kelas 0)\nPasien diindikasikan memiliki karakteristik yang masuk dalam kelompok risiko rendah.")

        m1, m2 = st.columns(2)
        with m1:
          st.metric(label="Parameter K yang Digunakan", value=f"K = {k_val}")
        with m2:
          st.metric(label="Tingkat Keyakinan Model", value=f"{max(probabilitas)*100:.1f}%")

        st.markdown("---")
        with st.expander(f"🔎 Lihat Detail {k_val} Tetangga Terdekat (Nearest Neighbors)", expanded=True):
          st.write(f"Daftar {k_val} data historis pasien terdekat yang menjadi acuan perhitungan jarak (Euclidean Distance):")
          
          # Membuat dataframe tetangga terdekat dan meletakkan Jarak_Euclidean di kolom paling akhir (setelah Kelas)
          df_tetangga = df_latih.iloc[indeks_tetangga[0]].copy()
          df_tetangga["Jarak_Euclidean"] = np.round(jarak[0], 4)
          
          st.dataframe(df_tetangga, use_container_width=True)
  else:
    st.info("👆 Silakan atur parameter dan isi formulir di atas, lalu klik tombol **Analisis Risiko Pasien**.")


# --- HALAMAN 2: ILUSTRASI GRAFIK KNN ---
elif menu == "Ilustrasi Grafik KNN":
  st.markdown('<p class="main-header">Ilustrasi Visual Algoritma KNN</p>', unsafe_allow_html=True)
  st.write("Visualisasi sebaran data pelatihan pasien berdasarkan atribut yang dapat dipilih bebas untuk melihat posisi titik data secara spasial.")

  col_f1, col_f2 = st.columns(2)
  with col_f1:
    fitur_x = st.selectbox(
        "Pilih Atribut Sumbu X:",
        ["Glukosa", "BMI", "Usia", "Tekanan_Darah"],
        index=0,
    )
  with col_f2:
    fitur_y = st.selectbox(
        "Pilih Atribut Sumbu Y:",
        ["BMI", "Glukosa", "Usia", "Tekanan_Darah"],
        index=0,
    )

  fig, ax = plt.subplots(figsize=(8, 5))

  df_rendah = df_latih[df_latih["Kelas"] == 0]
  df_tinggi = df_latih[df_latih["Kelas"] == 1]

  ax.scatter(df_rendah[fitur_x], df_rendah[fitur_y], color="#10B981", s=100, label="Risiko Rendah (0)", edgecolors="black")
  ax.scatter(df_tinggi[fitur_x], df_tinggi[fitur_y], color="#EF4444", s=100, label="Risiko Tinggi (1)", edgecolors="black")

  ax.set_xlabel(fitur_x.replace("_", " "), fontsize=12, fontweight="bold")
  ax.set_ylabel(fitur_y.replace("_", " "), fontsize=12, fontweight="bold")
  ax.set_title(f"Ilustrasi Sebaran Data Pasien ({fitur_x} vs {fitur_y})", fontsize=14, fontweight="bold", color="#1E3A8A")
  ax.grid(True, linestyle="--", alpha=0.6)
  ax.legend()

  st.pyplot(fig)
  st.info("💡 Grafik di atas menampilkan pemetaan sebaran data historis pasien berdasarkan kelas risikonya.")


# --- HALAMAN 3: EKSPLORASI DATASET ---
elif menu == "Eksplorasi Dataset":
  st.markdown('<p class="main-header">Eksplorasi Dataset Latih</p>', unsafe_allow_html=True)
  st.write("Berikut adalah data historis pasien yang dimuat langsung dari file **data-pasien.xlsx**:")
  st.dataframe(df_latih, use_container_width=True)
  
  st.markdown("### Ringkasan Statistik Data")
  st.dataframe(df_latih.describe(), use_container_width=True)


# --- HALAMAN 4: TENTANG METODE KNN (BAHASA SEDERHANA & MUDAH DIPAHAMI) ---
elif menu == "Tentang Metode KNN":
  st.markdown('<p class="main-header">Tentang Metode K-Nearest Neighbors (KNN)</p>', unsafe_allow_html=True)

  st.info(
      "💡 **Analogi Sederhana:** Bayangkan kamu pindah ke lingkungan baru"
      " dan ingin tahu apakah tetangga di sekitar ramah atau tidak. Cara"
      " paling gampang adalah dengan melihat **tetangga terdekat rumahmu**."
      " Jika sebagian besar tetangga terdekatmu ramah, kemungkinan besar kamu"
      " juga akan menganggap lingkungan itu ramah."
      " \n\nPrinsip inilah yang digunakan oleh **Algoritma KNN**!"
  )

  st.markdown("---")
  st.subheader("🤖 Bagaimana Cara Kerja Sistem Ini?")
  st.write(
      "Saat kamu memasukkan data pasien baru (seperti Usia, BMI, Glukosa, dan"
      " Tekanan Darah), komputer akan melakukan 3 langkah utama:"
  )

  st.markdown(
      "1. **Menghitung Jarak (Kemiripan):** Komputer membandingkan data pasien"
      " baru dengan seluruh data historis pasien yang ada di dalam file Excel"
      " menggunakan rumus jarak matematika (*Euclidean Distance*)."
  )
  st.markdown(
      "2. **Mencari Tetangga Terdekat ($K$):** Berdasarkan nilai $K$ yang kamu"
      " atur (misal $K=3$), komputer akan memilih 3 data pasien lama yang"
      " memiliki karakteristik **paling mirip (jaraknya paling kecil)** dengan"
      " pasien baru."
  )
  st.markdown(
      "3. **Keputusan Mayoritas (Voting):** Dari 3 pasien terdekat tadi, dilihat"
      " mana yang jumlahnya lebih banyak—apakah kelompok **Risiko Tinggi** atau"
      " **Risiko Rendah**. Kelompok terbanyak itulah yang akan menjadi hasil"
      " prediksi akhir!"
  )

  st.markdown("---")
  st.subheader("📐 Rumus Perhitungan Jarak (Euclidean Distance)")
  st.write(
      "Secara matematis, jarak kemiripan antar-pasien dihitung dengan rumus"
      " berikut:"
  )

  st.latex(r"d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}")

  st.markdown("Keterangan rumus:")
  st.markdown(
      "- $d(x, y)$ = Jarak kemiripan antara pasien baru ($x$) dan pasien lama"
      " ($y$)"
  )
  st.markdown(
      "- $(x_i - y_i)$ = Selisih nilai dari setiap parameter (misal selisih"
      " kadar glukosa atau BMI)"
  )
  st.markdown(
      "- $n$ = Total atribut yang dihitung (Usia, BMI, Glukosa, Tekanan Darah,"
      " Riwayat Keluarga)"
  )
