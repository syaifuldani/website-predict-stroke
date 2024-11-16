# Aplikasi Web Prediksi Stroke menggunakan KNN

Aplikasi web untuk memprediksi risiko stroke menggunakan algoritma K-Nearest Neighbors (KNN). Aplikasi ini dibangun menggunakan Streamlit dan Scikit-learn.

## 📋 Deskripsi

Aplikasi ini memungkinkan pengguna untuk:

-   Memahami data stroke melalui visualisasi dan analisis
-   Melihat proses preprocessing data
-   Memahami proses modeling dan evaluasi
-   Melakukan prediksi risiko stroke berdasarkan input yang diberikan

## 🔧 Teknologi yang Digunakan

-   Python 3.9+
-   Streamlit
-   Scikit-learn
-   Pandas
-   NumPy
-   Seaborn
-   Matplotlib

## 📦 Dependencies

```
streamlit==1.24.0
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
seaborn==0.12.2
matplotlib==3.7.1
```

## 💻 Instalasi & Menjalankan Aplikasi

1. Clone repository

```bash
git clone https://github.com/syaifuldani/stroke-prediction-app.git
cd stroke-prediction-app
```

2. Buat virtual environment (opsional tapi direkomendasikan)

```bash
python -m venv venv
```

3. Aktifkan virtual environment

-   Windows:

```bash
venv\Scripts\activate
```

-   Linux/Mac:

```bash
source venv/bin/activate
```

4. Install dependencies

```bash
pip install -r requirements.txt
```

5. Jalankan aplikasi

```bash
streamlit run app.py
```

## 📁 Struktur File

```
stroke-prediction-app/
├── main.py                     # File utama aplikasi
├── healthcare-dataset-stroke-data.csv    # Dataset
├── knn_model.pkl             # Model KNN yang sudah dilatih
├── scaler.pkl                # Scaler untuk normalisasi data
└── README.md                 # Dokumentasi
```

## 📊 Dataset

Dataset yang digunakan adalah Healthcare Dataset Stroke Data yang berisi informasi pasien seperti:

-   Gender
-   Age
-   Hypertension
-   Heart Disease
-   Ever Married
-   Work Type
-   Residence Type
-   Average Glucose Level
-   BMI
-   Smoking Status
-   Stroke (Target Variable)

## 🛠️ Fitur Aplikasi

1. **Data Understanding**

    - Visualisasi distribusi data
    - Analisis statistik deskriptif
    - Penjelasan setiap variabel

2. **Preprocessing**

    - Handling missing values
    - Label encoding untuk variabel kategorikal
    - Normalisasi data numerik
    - Penjelasan setiap tahap preprocessing

3. **Modelling & Evaluasi**

    - Implementasi algoritma KNN
    - Cross-validation
    - Evaluasi performa model
    - Learning curve analysis

4. **Prediksi**
    - Form input untuk data pasien
    - Hasil prediksi dengan probabilitas
    - Analisis faktor risiko
    - Rekomendasi berdasarkan hasil prediksi

## 📈 Performa Model

-   Accuracy: [X]%
-   Precision: [X]%
-   Recall: [X]%
-   F1-Score: [X]%

## 🚀 Cara Menggunakan Aplikasi

1. Buka aplikasi melalui browser
2. Pilih menu yang diinginkan dari sidebar
3. Untuk melakukan prediksi:
    - Pilih menu "Prediksi Stroke"
    - Isi semua informasi yang diminta
    - Klik tombol "Prediksi"
    - Hasil prediksi akan ditampilkan beserta analisis faktor risiko

## 🙏 Acknowledgments

-   [Healthcare Stroke Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
-   [Streamlit Documentation](https://docs.streamlit.io/)
-   [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---
