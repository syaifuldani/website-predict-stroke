import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="Prediksi Stroke",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        # Load best model berdasarkan hasil evaluasi
        best_model = pickle.load(open('knn_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return best_model, scaler
    except FileNotFoundError:
        st.error("Model belum dilatih. Silakan latih model terlebih dahulu di halaman Modeling & Evaluasi")
        return None, None

# Function to load the model and preprocessors
@st.cache_data
def load_data():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    # Hapus ID
    df = df.drop('id', axis=1)
    # df = df[df['smoking_status'] != 'Unknown']
    return df

def preprocess_input(data):
    # Membuat instance LabelEncoder baru untuk setiap kolom kategorikal
    # Gender
    le_gender = LabelEncoder()
    le_gender.fit(['Female', 'Male','Other'])
    data['gender'] = le_gender.transform(data['gender'])
    
    # Ever married
    le_married = LabelEncoder()
    le_married.fit(['No', 'Yes'])
    data['ever_married'] = le_married.transform(data['ever_married'])
    
    # Work type
    le_work = LabelEncoder()
    le_work.fit(['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children'])
    data['work_type'] = le_work.transform(data['work_type'])
    
    # Residence type
    le_residence = LabelEncoder()
    le_residence.fit(['Rural', 'Urban'])
    data['Residence_type'] = le_residence.transform(data['Residence_type'])
    
    # Smoking status
    le_smoking = LabelEncoder()
    le_smoking.fit(['formerly smoked', 'never smoked', 'smokes'])
    data['smoking_status'] = le_smoking.transform(data['smoking_status'])
    
    return data

def show_predict_page():
    st.title("Prediksi Stroke")
    st.write("Masukkan data pasien untuk memprediksi risiko stroke")
    
    # Tampilkan informasi dataset yang digunakan untuk training
    @st.cache_data
    def get_dataset_info():
        df = pd.read_csv('healthcare-dataset-stroke-data.csv')
        # Preprocessing seperti pada training
        # df = df[df['smoking_status'] != 'Unknown']
        df = df.drop('id', axis=1)
        return len(df)
    
    total_data = get_dataset_info()
    st.info(f"Model ini dilatih menggunakan {total_data} data yang telah dibersihkan")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox('Jenis Kelamin:', ['Female', 'Male','Other'])
            age = st.number_input('Usia:', min_value=0, max_value=120, value=30)
            hypertension = st.selectbox('Hipertensi:', ['0', '1'], help='0: Tidak, 1: Ya')
            heart_disease = st.selectbox('Penyakit Jantung:', ['0', '1'], help='0: Tidak, 1: Ya')
            ever_married = st.selectbox('Status Pernikahan:', ['No', 'Yes'])
        
        with col2:
            work_type = st.selectbox('Jenis Pekerjaan:', 
                                   ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
            Residence_type = st.selectbox('Tipe Tempat Tinggal:', ['Urban', 'Rural'])
            avg_glucose_level = st.number_input('Level Glukosa Rata-rata:', 
                                              min_value=0.0, max_value=300.0, value=90.0)
            bmi = st.number_input('BMI:', min_value=0.0, max_value=60.0, value=23.0)
            smoking_status = st.selectbox('Status Merokok:', 
                                        ['never smoked', 'formerly smoked', 'smokes','unknown'])

        submitted = st.form_submit_button("Prediksi")
        
        if submitted:
            input_data = {
                'gender': [gender],
                'age': [float(age)],
                'hypertension': [int(hypertension)],
                'heart_disease': [int(heart_disease)],
                'ever_married': [ever_married],
                'work_type': [work_type],
                'Residence_type': [Residence_type],
                'avg_glucose_level': [float(avg_glucose_level)],
                'bmi': [float(bmi)],
                'smoking_status': [smoking_status]
            }
            
            input_df = pd.DataFrame(input_data)
            
            # Load model dan scaler
            model, scaler = load_model()
            
            # Preprocessing input
            # Gender encoding: Female: 0, Male: 1
            le_gender = LabelEncoder()
            le_gender.fit(['Female', 'Male'])
            input_df['gender'] = le_gender.transform(input_df['gender'])
            
            # Ever married encoding: No: 0, Yes: 1
            le_married = LabelEncoder()
            le_married.fit(['No', 'Yes'])
            input_df['ever_married'] = le_married.transform(input_df['ever_married'])
            
            # Work type encoding
            le_work = LabelEncoder()
            le_work.fit(['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children'])
            input_df['work_type'] = le_work.transform(input_df['work_type'])
            
            # Residence type encoding: Rural: 0, Urban: 1
            le_residence = LabelEncoder()
            le_residence.fit(['Rural', 'Urban'])
            input_df['Residence_type'] = le_residence.transform(input_df['Residence_type'])
            
            # Smoking status encoding
            le_smoking = LabelEncoder()
            le_smoking.fit(['formerly smoked', 'never smoked', 'smokes','unknown'])
            input_df['smoking_status'] = le_smoking.transform(input_df['smoking_status'])
            
            # Scale numerical features
            numerical_cols = ['age', 'avg_glucose_level', 'bmi']
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            # Make prediction
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)
            
            # Display results
            st.subheader('Hasil Prediksi:')
            if prediction[0] == 1:
                st.error('⚠️ Pasien memiliki risiko stroke')
                st.write(f'Probabilitas terjadinya stroke: {probability[0][1]:.2%}')
            else:
                st.success('✅ Pasien tidak memiliki risiko stroke')
                st.write(f'Probabilitas tidak terjadinya stroke: {probability[0][0]:.2%}')
            
            # Analisis faktor risiko
            st.subheader('Analisis Faktor Risiko:')
            risk_factors = []
            
            if float(age) > 60:
                risk_factors.append("Usia di atas 60 tahun")
            if int(hypertension) == 1:
                risk_factors.append("Memiliki hipertensi")
            if int(heart_disease) == 1:
                risk_factors.append("Memiliki penyakit jantung")
            if float(avg_glucose_level) > 150:
                risk_factors.append("Level glukosa tinggi")
            if float(bmi) > 25:
                risk_factors.append("BMI di atas normal")
            if smoking_status == 'smokes':
                risk_factors.append("Perokok aktif")
                
            if risk_factors:
                st.warning("Faktor risiko yang teridentifikasi:")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("Tidak ada faktor risiko signifikan yang teridentifikasi")
            
            # Tampilkan interpretasi hasil
            st.subheader("Interpretasi Hasil:")
            
            confidence = max(probability[0]) * 100
            st.write(f"Tingkat kepercayaan prediksi: {confidence:.1f}%")
            
            if confidence < 60:
                st.warning("⚠️ Prediksi ini memiliki tingkat kepercayaan rendah. Disarankan untuk konsultasi dengan profesional kesehatan.")
            
            # Rekomendasi
            st.subheader("Rekomendasi:")
            if prediction[0] == 1:
                st.markdown("""
                1. Segera konsultasikan dengan dokter
                2. Lakukan pemeriksaan kesehatan rutin
                3. Jaga pola hidup sehat
                4. Kontrol faktor risiko yang teridentifikasi
                """)
            else:
                st.markdown("""
                1. Tetap jaga pola hidup sehat
                2. Lakukan pemeriksaan kesehatan secara berkala
                3. Hindari faktor risiko stroke
                """)
            

def show_data_page():
    st.title("Data Stroke Prediction")
    
    # Load dataset
    @st.cache_data
    def load_data():
        # Load data asli
        df = pd.read_csv('healthcare-dataset-stroke-data.csv')
        return df
    
    data = load_data()
    
    # Tampilkan dataset asli
    st.subheader("Dataset Original")
    st.write("Raw data sebelum preprocessing:")
    st.dataframe(data)
    
    # Informasi ukuran dataset original
    st.write(f"Jumlah data original: {data.shape[0]} baris")
    st.write(f"Jumlah fitur: {data.shape[1]} kolom")
    
    # Proses data cleaning
    data_cleaned = data.copy()
    # Hapus kolom ID
    data_cleaned = data_cleaned.drop('id', axis=1)
    # Hapus kategori 'Unknown' di smoking_status
    # data_cleaned = data_cleaned[data_cleaned['smoking_status'] != 'Unknown']
    
    # Tampilkan dataset setelah cleaning
    st.subheader("Dataset Setelah Cleaning")
    st.write("Data setelah menghapus kategori 'Unknown' pada smoking_status:")
    st.dataframe(data_cleaned)
    st.write(f"Jumlah data setelah cleaning: {data_cleaned.shape[0]} baris")
    
    # Penjelasan Atribut
    st.subheader("Penjelasan Atribut Dataset")
    
    # Gunakan expander untuk membuat tampilan lebih rapi
    with st.expander("Lihat Penjelasan Detail Atribut"):
        st.markdown("""
        1. **gender**: Jenis kelamin pasien
            - Male
            - Female
            - Other
        
        2. **age**: Usia pasien
        
        3. **hypertension**: Status hipertensi
            - 0: Pasien tidak memiliki hipertensi
            - 1: Pasien memiliki hipertensi
        
        4. **heart_disease**: Status penyakit jantung
            - 0: Pasien tidak memiliki penyakit jantung
            - 1: Pasien memiliki penyakit jantung
        
        5. **ever_married**: Status pernikahan
            - No
            - Yes
        
        6. **work_type**: Jenis pekerjaan
            - Children
            - Govt_job
            - Never_worked
            - Private
            - Self-employed
        
        7. **Residence_type**: Tipe tempat tinggal
            - Rural
            - Urban
        
        8. **avg_glucose_level**: Level glukosa rata-rata dalam darah
        
        9. **bmi**: Body Mass Index (Indeks Massa Tubuh)
        
        10. **smoking_status**: Status merokok
            - formerly smoked: Pernah merokok
            - never smoked: Tidak pernah merokok
            - smokes: Perokok aktif
        
        11. **stroke**: Target variable (Label)
            - 0: Pasien tidak mengalami stroke
            - 1: Pasien mengalami stroke
        """)
    
    # Tampilkan informasi statistik dasar
    st.subheader("Informasi Statistik Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Statistik numerik:")
        st.write(data_cleaned.describe())
    
    with col2:
        st.write("Informasi tipe data:")
        st.write(data_cleaned.dtypes)
    
    # Tampilkan distribusi untuk setiap kategori
    st.subheader("Distribusi Data Kategorikal")
    
    # Gender distribution
    col1, col2 = st.columns(2)
    with col1:
        st.write("Distribusi Gender:")
        st.write(data_cleaned['gender'].value_counts())
        
        # Visualisasi gender
        fig_gender = plt.figure(figsize=(8, 6))
        sns.countplot(data=data_cleaned, x='gender')
        plt.title('Distribusi Gender')
        st.pyplot(fig_gender)
    
    with col2:
        st.write("Distribusi Smoking Status:")
        st.write(data_cleaned['smoking_status'].value_counts())
        
        # Visualisasi smoking status
        fig_smoking = plt.figure(figsize=(8, 6))
        sns.countplot(data=data_cleaned, x='smoking_status')
        plt.xticks(rotation=45)
        plt.title('Distribusi Smoking Status')
        st.pyplot(fig_smoking)
    
    # Tampilkan distribusi kelas stroke
    st.subheader("Distribusi Kelas Stroke")
    stroke_dist = data_cleaned['stroke'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Jumlah kasus per kelas:")
        st.write(f"- Tidak Stroke (0): {stroke_dist[0]} kasus")
        st.write(f"- Stroke (1): {stroke_dist[1]} kasus")
        
    with col2:
        # Visualisasi distribusi stroke
        fig_stroke = plt.figure(figsize=(8, 6))
        plt.pie(stroke_dist, labels=['Tidak Stroke', 'Stroke'], autopct='%1.1f%%', 
                colors=['lightblue', 'lightcoral'])
        plt.title('Distribusi Kasus Stroke')
        st.pyplot(fig_stroke)
    
    # Tampilkan informasi missing values
    st.subheader("Informasi Missing Values")
    missing_values = data_cleaned.isnull().sum()
    if missing_values.any():
        st.write("Jumlah missing values per kolom:")
        st.write(missing_values[missing_values > 0])
    else:
        st.write("Tidak ada missing values dalam dataset yang sudah dibersihkan")

def show_preprocessing_page():
    st.title("Data Preprocessing")
    
    # Load dataset
    @st.cache_data
    def load_data():
        return pd.read_csv('healthcare-dataset-stroke-data.csv')
    
    data = load_data()
    
    # 1. Cleaning Data
    st.header("1. Cleaning Data")
    st.write("Tahap awal preprocessing adalah membersihkan data dari atribut yang tidak diperlukan.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Data sebelum cleaning:")
        st.write(f"- Jumlah baris: {data.shape[0]}")
        st.write(f"- Jumlah kolom: {data.shape[1]}")
        st.write("- Distribusi Gender:")
        st.write(data['gender'].value_counts())
        st.write("- Distribusi Smoking Status:")
        st.write(data['smoking_status'].value_counts())
    
    # Hapus kolom ID
    data = data.drop('id', axis=1)
    # Hapus kategori 'Unknown' di smoking_status
    # data = data[data['smoking_status'] != 'Unknown']
    
    with col2:
        st.write("Data setelah cleaning:")
        st.write(f"- Jumlah baris: {data.shape[0]}")
        st.write(f"- Jumlah kolom: {data.shape[1]}")
        st.write("- Distribusi Gender setelah cleaning:")
        st.write(data['gender'].value_counts())
        st.write("- Distribusi Smoking Status setelah cleaning:")
        st.write(data['smoking_status'].value_counts())
    
    # 2. Handling Missing Value
    st.header("2. Handling Missing Value")
    st.write("Memeriksa dan menangani missing value dalam dataset.")
    
    # Tampilkan jumlah missing value per kolom
    missing_values = data.isnull().sum()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Jumlah missing value per kolom:")
        st.write(missing_values)
        
        # Visualisasi missing value
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_values.plot(kind='bar')
        plt.title('Jumlah Missing Value per Kolom')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Handle missing values in BMI using median
    bmi_median = data['bmi'].median()
    data['bmi'] = data['bmi'].fillna(bmi_median)
    
    st.write("""
    1. Kolom BMI:
       - Menggunakan nilai median untuk mengisi missing value
       - Nilai median BMI: {:.2f}
    """.format(bmi_median))
    
    # 4. Konversi Data Kategorik ke Numerik
    # Di bagian 4. Konversi Data Kategorik ke Numerik, tambahkan penjelasan berikut:

    st.header("4. Konversi Data Kategorik ke Numerik")
    st.write("Mengubah variabel kategorik menjadi numerik menggunakan Label Encoding.")
    
    # Tampilkan contoh data sebelum encoding
    st.subheader("Data Kategorik Sebelum Encoding:")
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    st.write(data[categorical_columns].head())
    
    # Lakukan label encoding
    le = LabelEncoder()
    data_encoded = data.copy()
    for column in categorical_columns:
        data_encoded[column] = le.fit_transform(data_encoded[column])
    
    # Tampilkan hasil encoding
    st.subheader("Setelah Label Encoding:")
    st.write(data_encoded[categorical_columns].head())
    
    # Tampilkan mapping untuk setiap kolom kategorikal
    st.subheader("Mapping Label Encoding:")
    for column in categorical_columns:
        le = LabelEncoder()
        le.fit(data[column])
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        st.write(f"\nMapping untuk kolom {column}:")
        for original, encoded in mapping.items():
            st.write(f"- {original}: {encoded}")

    # Tambahkan ini setelah bagian Label Encoding di halaman preprocessing
    st.header("Penjelasan Label Encoding")

    with st.expander("Klik untuk melihat penjelasan detail tentang Label Encoding"):
        st.markdown("""
        ### Tentang Penempatan Nilai pada Label Encoding

        Label Encoding bekerja dengan mengubah nilai kategorikal menjadi numerik berdasarkan urutan alfabetis. 
        Ini adalah proses teknis murni, bukan berdasarkan preferensi atau bias tertentu.

        #### Bagaimana Label Encoding Bekerja:
            1. Label Encoder akan mengurutkan nilai unik dalam data secara alfabetis
            2. Kemudian memberikan nilai numerik mulai dari 0 sesuai urutan tersebut
            3. Proses ini bersifat otomatis dan konsisten

        #### Contoh pada Atribut Gender:
            - 'Female' → 0 (karena 'F' muncul lebih dulu dalam alfabet)
            - 'Male' → 1 (karena 'M' muncul setelah 'F')

        #### Penting untuk Diketahui:
            1. Penempatan nilai 0 dan 1 adalah hasil dari pengurutan alfabetis, bukan karena pertimbangan khusus atau bias gender
            2. Jika data gender ditulis 'Pria' dan 'Wanita', maka:
                - 'Pria' → 0 (karena 'P' lebih dulu)
                - 'Wanita' → 1 (karena 'W' setelah 'P')

        #### Mengapa Tidak Ada Jurnal Khusus:
        Tidak ada jurnal khusus yang membahas penempatan nilai 0 dan 1 untuk gender karena:
            1. Ini adalah proses teknis dalam pengolahan data
            2. Urutan nilai tidak mempengaruhi performa model
            3. Nilai yang diberikan bersifat arbitrer dan tidak memiliki makna hierarkis

        #### Contoh pada Atribut Lain:
            - Ever_married: No(0), Yes(1) - karena N sebelum Y
            - Residence_type: Rural(0), Urban(1) - karena R sebelum U
            - Work_type: Govt_job(0), Never_worked(1), Private(2), Self-employed(3), children(4) - urutan alfabetis

        #### Alternative Encoding Methods:
        Jika urutan atau hubungan antar kategori penting, bisa menggunakan metode encoding lain:
            1. One-Hot Encoding
            2. Ordinal Encoding
            3. Binary Encoding
            4. Target Encoding

        #### Kesimpulan:
            - Label Encoding adalah proses teknis berbasis alfabet
            - Nilai 0 dan 1 tidak memiliki arti khusus
            - Urutan berbasis alfabet memastikan konsistensi dalam pengolahan data
            - Pemilihan metode encoding harus disesuaikan dengan karakteristik data dan kebutuhan model

        #### Tips Penggunaan:
            1. Dokumentasikan proses encoding dengan baik
            2. Pastikan konsistensi encoding antara data training dan testing
            3. Pertimbangkan karakteristik data dalam memilih metode encoding
            4. Jika urutan kategori penting, gunakan metode encoding yang sesuai

        #### Referensi Teknis:
            1. [Scikit-learn Documentation - LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
            2. [Feature Encoding Techniques - Towards Data Science](https://towardsdatascience.com/feature-encoding-techniques-a-comprehensive-study-part-1-label-encoding-12da1c345aa4)
        """)

        # Tampilkan contoh sederhana
        st.subheader("Contoh Sederhana Label Encoding:")
        example_df = pd.DataFrame({
            'Kategori': ['Female', 'Male', 'Female', 'Male'],
            'Hasil Encoding': [0, 1, 0, 1]
        })
        st.table(example_df)

    # 5. Pemisahan Target
    st.header("5. Pemisahan Target")
    st.write("Memisahkan atribut target (stroke) dari dataset.")
    
    X = data_encoded.drop('stroke', axis=1)
    y = data_encoded['stroke']
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Features (X):")
        st.write(X.head())
        st.write(f"Shape: {X.shape}")
    
    with col2:
        st.write("Target (y):")
        st.write(pd.DataFrame(y.head(), columns=['stroke']))
        st.write(f"Shape: {y.shape}")
    
    # 6. Normalisasi Data
    st.header("6. Normalisasi Data")
    st.write("Melakukan normalisasi pada fitur numerik menggunakan Min-Max Scaler.")
    
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    
    # Tampilkan data sebelum normalisasi
    st.subheader("Sebelum Normalisasi:")
    st.write(data_encoded[numerical_cols].describe())
    
    # Lakukan normalisasi
    scaler = MinMaxScaler()
    data_encoded[numerical_cols] = scaler.fit_transform(data_encoded[numerical_cols])
    
    # Tampilkan data setelah normalisasi
    st.subheader("Setelah Normalisasi:")
    st.write(data_encoded[numerical_cols].describe())
    
    # Visualisasi distribusi setelah normalisasi
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(numerical_cols):
        sns.histplot(data=data_encoded, x=col, ax=axes[i])
        axes[i].set_title(f'Distribusi {col} setelah Normalisasi')
    plt.tight_layout()
    st.pyplot(fig)

    # 7. SMOTE (Synthetic Minority Over-sampling Technique)
    st.header("7. Penanganan Imbalanced Data dengan SMOTE")
    st.write("Melakukan oversampling pada kelas minoritas menggunakan teknik SMOTE.")

    # Import SMOTE
    from imblearn.over_sampling import SMOTE
    from collections import Counter

    # Tampilkan distribusi kelas sebelum SMOTE
    st.subheader("Distribusi Kelas Sebelum SMOTE:")
    class_dist_before = pd.Series(y).value_counts()
    st.write("Jumlah sampel per kelas:")
    st.write(class_dist_before)

    # Visualisasi distribusi sebelum SMOTE
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    plt.pie(class_dist_before.values, 
            labels=['Tidak Stroke', 'Stroke'], 
            autopct='%1.1f%%',
            colors=['lightblue', 'lightcoral'])
    plt.title('Distribusi Kelas Sebelum SMOTE')
    st.pyplot(fig1)

    # Terapkan SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Tampilkan distribusi kelas setelah SMOTE
    st.subheader("Distribusi Kelas Setelah SMOTE:")
    class_dist_after = pd.Series(y_resampled).value_counts()
    st.write("Jumlah sampel per kelas:")
    st.write(class_dist_after)

    # Visualisasi distribusi setelah SMOTE
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plt.pie(class_dist_after.values, 
            labels=['Tidak Stroke', 'Stroke'], 
            autopct='%1.1f%%',
            colors=['lightblue', 'lightcoral'])
    plt.title('Distribusi Kelas Setelah SMOTE')
    st.pyplot(fig2)

    # Perbandingan jumlah sampel
    st.subheader("Perbandingan Jumlah Sampel:")
    comparison_df = pd.DataFrame({
        'Sebelum SMOTE': class_dist_before,
        'Setelah SMOTE': class_dist_after
    })
    st.write(comparison_df)

    # Visualisasi perbandingan
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    comparison_df.plot(kind='bar', ax=ax3)
    plt.title('Perbandingan Jumlah Sampel Sebelum dan Setelah SMOTE')
    plt.xlabel('Kelas')
    plt.ylabel('Jumlah Sampel')
    plt.xticks(rotation=45)
    plt.legend(title='')
    plt.tight_layout()
    st.pyplot(fig3)

    # Penjelasan SMOTE
    with st.expander("Penjelasan Teknik SMOTE"):
        st.markdown("""
        ### SMOTE (Synthetic Minority Over-sampling Technique)
        
        SMOTE adalah teknik untuk menangani masalah ketidakseimbangan kelas (imbalanced class) dengan cara membuat sampel sintetis dari kelas minoritas.
        
        #### Cara Kerja SMOTE:
        1. Mengidentifikasi kelas minoritas
        2. Untuk setiap sampel di kelas minoritas:
            - Mencari k-nearest neighbors
            - Memilih secara acak satu dari k-nearest neighbors
            - Membuat sampel sintetis di antara sampel asli dan neighbor yang dipilih
        
        #### Keuntungan SMOTE:
        - Mengurangi risiko overfitting dibandingkan random oversampling
        - Menghasilkan sampel sintetis yang masuk akal
        - Meningkatkan performa model pada kelas minoritas
        
        #### Kekurangan SMOTE:
        - Dapat menghasilkan noise pada data
        - Membutuhkan waktu komputasi lebih lama
        - Perlu berhati-hati dalam menentukan jumlah sampel sintetis
        
        #### Parameter Penting:
        - sampling_strategy: menentukan rasio oversampling
        - k_neighbors: jumlah tetangga terdekat yang digunakan
        - random_state: untuk reproducibility
        """)


def show_modeling_page():
    st.title("Modeling & Evaluasi")
    st.write("Perbandingan beberapa metode klasifikasi untuk prediksi stroke")
    
    # Load dan preprocessing data
    @st.cache_data
    def load_data():
        df = pd.read_csv('healthcare-dataset-stroke-data.csv')
        df = df.drop('id', axis=1)
        # df = df[df['smoking_status'] != 'Unknown']
        return df
    
    data = load_data()
    
    # 1. Data Preparation
    st.header("1. Persiapan Data")
    
    # Handling missing values
    data['bmi'].fillna(data['bmi'].median(), inplace=True)
    
    # Label Encoding
    le = LabelEncoder()
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    
    # Pemisahan fitur dan target
    X = data.drop('stroke', axis=1)
    y = data['stroke']
    
    # Normalisasi
    scaler = MinMaxScaler()
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # 2. Train Test Split
    st.header("2. Pembagian Data Training dan Testing")
    test_size = st.slider("Pilih rasio data testing (%):", 10, 40, 20)
    test_size = test_size / 100
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Jumlah data training:", len(X_train))
        st.write("Distribusi kelas pada data training:")
        st.write(pd.Series(y_train).value_counts())
    
    with col2:
        st.write("Jumlah data testing:", len(X_test))
        st.write("Distribusi kelas pada data testing:")
        st.write(pd.Series(y_test).value_counts())

    # 3. SMOTE Implementation
    st.header("3. Penerapan SMOTE")
    st.write("Menyeimbangkan distribusi kelas menggunakan teknik SMOTE")

    # Tampilkan distribusi kelas sebelum SMOTE
    col1, col2 = st.columns(2)
    with col1:
        st.write("Distribusi kelas sebelum SMOTE:")
        original_dist = pd.Series(y_train).value_counts()
        st.write(original_dist)
        
        # Visualisasi pie chart sebelum SMOTE
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        plt.pie(original_dist.values, 
                labels=['Tidak Stroke', 'Stroke'], 
                autopct='%1.1f%%',
                colors=['lightblue', 'lightcoral'])
        plt.title('Distribusi Kelas Sebelum SMOTE')
        st.pyplot(fig1)

    # Terapkan SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    with col2:
        st.write("Distribusi kelas setelah SMOTE:")
        balanced_dist = pd.Series(y_train_balanced).value_counts()
        st.write(balanced_dist)
        
        # Visualisasi pie chart setelah SMOTE
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        plt.pie(balanced_dist.values, 
                labels=['Tidak Stroke', 'Stroke'], 
                autopct='%1.1f%%',
                colors=['lightblue', 'lightcoral'])
        plt.title('Distribusi Kelas Setelah SMOTE')
        st.pyplot(fig2)

    # 4. Model Training and Evaluation
    st.header("4. Pelatihan dan Evaluasi Model")

    # Tambahkan input untuk parameter KNN
    st.subheader("Parameter Model KNN")
    n_neighbors = st.number_input("Masukkan nilai K (jumlah tetangga):", min_value=1, max_value=50, value=5)
    
    # Function to evaluate model
    def evaluate_model(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        
        # Test metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Training metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        
        return accuracy, precision, recall, f1, cm, y_pred, train_accuracy

    # Dictionary of models
    models = {
        'KNN': KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='distance',
            metric='manhattan'  
        ),
        'SVM': SVC(
            probability=True,
            kernel='rbf',
            C=1.0,
            class_weight='balanced'
        ),
        'Naive Bayes': GaussianNB()
    }

    # Train and evaluate all models with balanced data
    results = {}
    for name, model in models.items():
        with st.spinner(f'Training {name} model menggunakan data yang sudah diseimbangkan...'):
            accuracy, precision, recall, f1, cm, y_pred, train_accuracy = evaluate_model(
                model, X_train_balanced, X_test, y_train_balanced, y_test
            )
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'train_accuracy': train_accuracy
            }
            
            # Save model
            with open(f'{name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
                pickle.dump(model, f)

    # 4. Display Results
    st.header("5. Perbandingan Hasil Model")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': results.keys(),
        'Training Accuracy': [results[model]['train_accuracy'] for model in results],
        'Testing Accuracy': [results[model]['accuracy'] for model in results],
        'Precision': [results[model]['precision'] for model in results],
        'Recall': [results[model]['recall'] for model in results],
        'F1-Score': [results[model]['f1'] for model in results]
    })
    
    # Display metrics comparison
    st.subheader("Metrik Perbandingan")
    st.dataframe(comparison_df.style.highlight_max(axis=0))
    
    # Plot metrics comparison
    fig = plt.figure(figsize=(12, 6))
    metrics = ['Training Accuracy', 'Testing Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(models))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, comparison_df[metric], width, label=metric)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width*2, comparison_df['Model'], rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display confusion matrices
    st.subheader("Confusion Matrices")
    cols = st.columns(len(models))
    for i, (name, model_results) in enumerate(results.items()):
        with cols[i]:
            st.write(f"{name}")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(model_results['confusion_matrix'], 
                       annot=True, 
                       fmt='d', 
                       cmap='Blues',
                       ax=ax)
            plt.title(f'{name}\nConfusion Matrix')
            st.pyplot(fig)

    # 5. K-Fold Cross Validation for all models
    st.header("5. K-Fold Cross Validation")
    
    n_folds = st.number_input("Masukkan jumlah fold:", min_value=2, max_value=10, value=5)
    
    from sklearn.model_selection import cross_val_score
    
    cv_results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=n_folds)
        cv_results[name] = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
        
        st.write(f"\n**{name}**")
        st.write(f"Scores untuk setiap fold:")
        for i, score in enumerate(cv_scores, 1):
            st.write(f"Fold {i}: {score:.3f}")
        st.write(f"Rata-rata accuracy: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")

    # 6. Learning Curves
    st.header("6. Learning Curves")
    
    from sklearn.model_selection import learning_curve
    
    def plot_learning_curve(model, title):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        plt.fill_between(train_sizes, train_mean - train_std, 
                        train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, 
                        test_mean + test_std, alpha=0.1)
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title(f'Learning Curve - {title}')
        plt.legend(loc='best')
        plt.grid(True)
        return fig

    # Plot learning curves for all models
    for name, model in models.items():
        st.subheader(f"Learning Curve - {name}")
        fig = plot_learning_curve(model, name)
        st.pyplot(fig)
        
        # Analisis learning curve
        with st.expander(f"Analisis Learning Curve - {name}"):
            st.markdown("""
            ### Komponen Learning Curve:
            
            1. **Garis Training Score**:
               - Menunjukkan performa model pada data training
               - Semakin tinggi nilai, semakin baik model mempelajari pola data
               
            2. **Garis Cross-validation Score**:
               - Menunjukkan performa model pada data validasi
               - Mengindikasikan kemampuan generalisasi model
               
            3. **Area Bayangan**:
               - Menunjukkan standar deviasi scores
               - Semakin sempit area, semakin stabil performa model
            """)

    # 7. Best Model Analysis
    st.header("7. Analisis Model Terbaik")
    best_model = comparison_df.loc[comparison_df['Testing Accuracy'].idxmax(), 'Model']
    best_accuracy = comparison_df['Testing Accuracy'].max()
    
    st.write(f"""
    **Model Terbaik: {best_model}**
    - Training Accuracy: {results[best_model]['train_accuracy']:.3f}
    - Testing Accuracy: {results[best_model]['accuracy']:.3f}
    - Precision: {results[best_model]['precision']:.3f}
    - Recall: {results[best_model]['recall']:.3f}
    - F1-Score: {results[best_model]['f1']:.3f}
    """)
    
    # 8. Model Characteristics
    st.header("8. Karakteristik Model")
    st.write("""
    **1. K-Nearest Neighbors (KNN)**
    - Model non-parametrik yang sederhana
    - Bekerja baik dengan dataset kecil-menengah
    - Sensitif terhadap fitur yang tidak relevan
    
    **2. Naive Bayes**
    - Cepat dan efisien
    - Bekerja baik dengan data kategorikal
    - Mengasumsikan independensi antar fitur
    
    **2. Support Vector Machine (SVM)**
    - Efektif untuk data dimensi tinggi
    - Menggunakan kernel RBF untuk menangani non-linearitas
    - Class weight 'balanced' untuk menangani imbalanced data
    - Parameter C=1.0 untuk optimasi margin
    """)
    
    # 9. Save Models
    st.header("9. Penyimpanan Model")
    best_model_name = best_model.lower().replace(" ", "_")
    st.write(f"Model terbaik ({best_model}) telah disimpan sebagai '{best_model_name}_model.pkl'")
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    st.write("Scaler telah disimpan sebagai 'scaler.pkl'")

    # 10. Kesimpulan
    st.header("10. Kesimpulan")
    
    # Calculate gaps
    gaps = comparison_df['Training Accuracy'] - comparison_df['Testing Accuracy']
    
    st.write(f"""
    Berdasarkan hasil evaluasi di atas, beberapa kesimpulan yang dapat diambil:
    
    1. **Perbandingan Model:**
       - Model terbaik adalah {best_model} dengan accuracy {best_accuracy:.3f}
       - Gap terkecil antara training dan testing accuracy adalah {gaps.min():.3f} ({comparison_df.iloc[gaps.argmin()]['Model']})
       
    2. **Stabilitas Model:**
       - Cross-validation menunjukkan {best_model} memiliki performa yang paling stabil
       - Standar deviasi CV score: {cv_results[best_model]['std']:.3f}
       
    3. **Karakteristik Data:**
       - Dataset menunjukkan ketidakseimbangan kelas
       - Performa model dipengaruhi oleh distribusi kelas
       
    4. **Rekomendasi:**
       - Gunakan {best_model} untuk implementasi
       - Pertimbangkan teknik balancing data untuk meningkatkan performa
       - Monitor performa model secara berkala
    """)

def main():
    st.sidebar.title("Menu Bar")    
    
    # Add a logo or image at the top of the sidebar (optional)
    # st.sidebar.image("logo.png", width=100)
    
    # Create the navigation menu
    page = st.sidebar.radio(
        "Pilih Menu:",
        ["Data Understanding", "Preprocessing", "Modelling & Evaluasi", "Prediksi Stroke"]
    )
    
    # Additional info in sidebar
    st.sidebar.info(
        "Aplikasi ini menggunakan metode KNN untuk "
        "memprediksi risiko stroke berdasarkan berbagai faktor kesehatan."
    )
    
    # Show the selected page
    if page == "Data Understanding":
        show_data_page()
    elif page == "Preprocessing":
        show_preprocessing_page()
    elif page == "Modelling & Evaluasi":
        show_modeling_page()
    else:
        show_predict_page()

if __name__ == '__main__':
    main()