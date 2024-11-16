import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Prediksi Stroke",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = pickle.load(open('knn_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

# Function to load the model and preprocessors
@st.cache_data
def load_data():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    # Hapus ID
    df = df.drop('id', axis=1)
    # Hapus kategori 'Other' di gender dan 'Unknown' di smoking_status
    df = df[df['gender'] != 'Other']
    df = df[df['smoking_status'] != 'Unknown']
    return df

def preprocess_input(data):
    # Membuat instance LabelEncoder baru untuk setiap kolom kategorikal
    # Gender
    le_gender = LabelEncoder()
    le_gender.fit(['Female', 'Male'])
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
        df = df[df['gender'] != 'Other']
        df = df[df['smoking_status'] != 'Unknown']
        df = df.drop('id', axis=1)
        return len(df)
    
    total_data = get_dataset_info()
    st.info(f"Model ini dilatih menggunakan {total_data} data yang telah dibersihkan (tanpa kategori 'Other' pada gender dan 'Unknown' pada smoking status)")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox('Jenis Kelamin:', ['Female', 'Male'])  # Removed 'Other'
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
                                        ['never smoked', 'formerly smoked', 'smokes'])  # Removed 'Unknown'

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
            le_smoking.fit(['formerly smoked', 'never smoked', 'smokes'])
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
    # Hapus kategori 'Other' di gender
    data_cleaned = data_cleaned[data_cleaned['gender'] != 'Other']
    # Hapus kategori 'Unknown' di smoking_status
    data_cleaned = data_cleaned[data_cleaned['smoking_status'] != 'Unknown']
    
    # Tampilkan dataset setelah cleaning
    st.subheader("Dataset Setelah Cleaning")
    st.write("Data setelah menghapus kategori 'Other' pada gender dan 'Unknown' pada smoking_status:")
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
    # Hapus kategori 'Other' di gender
    data = data[data['gender'] != 'Other']
    # Hapus kategori 'Unknown' di smoking_status
    data = data[data['smoking_status'] != 'Unknown']
    
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
    
    # Ringkasan Preprocessing
    st.header("Ringkasan Preprocessing")
    st.write("""
    1. **Cleaning Data**:
       - Menghapus kolom ID yang tidak diperlukan
       
    2. **Handling Missing Value**:
       - Mengidentifikasi missing value pada dataset
       - Mengisi missing value pada kolom BMI dengan nilai median
       
    3. **Konversi Data Kategorik**:
       - Mengubah variabel kategorik menjadi numerik menggunakan Label Encoding
       - Kolom yang diubah: gender, ever_married, work_type, Residence_type, smoking_status
       
    4. **Pemisahan Target**:
       - Memisahkan kolom stroke sebagai target
       - Menyiapkan features dan target untuk modeling
       
    5. **Normalisasi Data**:
       - Menormalisasi fitur numerik (age, avg_glucose_level, bmi)
       - Menggunakan Min-Max Scaler untuk transformasi ke range [0,1]
    """)

def show_modeling_page():
    st.title("Modeling & Evaluasi")
    st.write("Proses pemodelan menggunakan algoritma K-Nearest Neighbors (KNN)")
    
    # Load dan preprocessing data
    @st.cache_data
    def load_data():
        df = pd.read_csv('healthcare-dataset-stroke-data.csv')
        df = df.drop('id', axis=1)
        df = df[df['gender'] != 'Other']
        df = df[df['smoking_status'] != 'Unknown']
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
    
    # Tambahkan slider untuk memilih rasio pembagian data
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
    
    # 3. Model Training
    st.header("3. Pelatihan Model")
    
    # Tambahkan input untuk parameter KNN
    n_neighbors = st.number_input("Masukkan nilai K (jumlah tetangga):", min_value=1, max_value=50, value=5)
    
    # Training model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    
    # Save model
    with open('knn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # 4. Model Evaluation
    st.header("4. Evaluasi Model")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # Function to calculate and display metrics
    def display_metrics(y_true, y_pred, dataset_name):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        st.write(f"Metrics untuk {dataset_name}:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{acc:.3f}")
        with col2:
            st.metric("Precision", f"{prec:.3f}")
        with col3:
            st.metric("Recall", f"{rec:.3f}")
        with col4:
            st.metric("F1-Score", f"{f1:.3f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)
    
    # Display metrics for both training and testing data
    st.subheader("Metrics pada Data Training")
    display_metrics(y_train, y_pred_train, "Data Training")
    
    st.subheader("Metrics pada Data Testing")
    display_metrics(y_test, y_pred, "Data Testing")
    
    # 5. K-Fold Cross Validation
    st.header("5. K-Fold Cross Validation")
    
    from sklearn.model_selection import cross_val_score
    n_folds = st.number_input("Masukkan jumlah fold:", min_value=2, max_value=10, value=5)
    
    cv_scores = cross_val_score(model, X, y, cv=n_folds)
    
    st.write(f"Scores untuk setiap fold:")
    for i, score in enumerate(cv_scores, 1):
        st.write(f"Fold {i}: {score:.3f}")
    
    st.write(f"Rata-rata accuracy: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
    
    # 6. Learning Curve
    # 6. Learning Curve
    st.header("6. Learning Curve")
    
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue')
    plt.plot(train_sizes, test_mean, label='Cross-validation score', color='green')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')
    plt.xlabel('Jumlah Data Training')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True)
    st.pyplot(fig)

    # Tambahkan penjelasan learning curve
    st.subheader("Interpretasi Learning Curve")
    
    # Analisis gap antara training dan validation score
    gap = np.mean(train_mean - test_mean)
    
    # Analisis tren validation score
    validation_trend = test_mean[-1] - test_mean[0]
    
    # Deteksi overfitting/underfitting
    if gap > 0.1:  # threshold bisa disesuaikan
        fitting_status = "overfitting"
    elif train_mean[-1] < 0.8:  # threshold bisa disesuaikan
        fitting_status = "underfitting"
    else:
        fitting_status = "good fit"

    with st.expander("Klik untuk melihat penjelasan detail Learning Curve"):
        st.markdown("""
        ### Komponen Learning Curve:
        
        1. **Garis Biru (Training Score)**:
           - Menunjukkan performa model pada data training
           - Semakin tinggi nilai, semakin baik model mempelajari pola data training
           
        2. **Garis Hijau (Cross-validation Score)**:
           - Menunjukkan performa model pada data validasi
           - Mengindikasikan kemampuan generalisasi model
           
        3. **Area Bayangan**:
           - Menunjukkan standar deviasi scores
           - Semakin sempit area, semakin stabil performa model
        
        ### Analisis Grafik:
        
        1. **Gap antara Training dan Validation**:
           - Gap saat ini: {:.3f}
           - Gap kecil (< 0.1): Model memiliki variance rendah
           - Gap besar (> 0.1): Mengindikasikan overfitting
        
        2. **Tren Validation Score**:
           - Perubahan dari awal ke akhir: {:.3f}
           - Positif: Model membaik dengan penambahan data
           - Negatif: Model mungkin memerlukan penyesuaian
        
        3. **Status Fitting Model**: {}
        
        ### Rekomendasi berdasarkan grafik:
        """.format(gap, validation_trend, fitting_status))

        # Berikan rekomendasi berdasarkan analisis
        if fitting_status == "overfitting":
            st.markdown("""
            * **Tindakan yang disarankan untuk mengatasi overfitting:**
                1. Kurangi kompleksitas model dengan mengurangi nilai K
                2. Tambah data training jika memungkinkan
                3. Pertimbangkan feature selection
                4. Lakukan regularisasi jika menggunakan model lain
            """)
        elif fitting_status == "underfitting":
            st.markdown("""
            * **Tindakan yang disarankan untuk mengatasi underfitting:**
                1. Tingkatkan kompleksitas model dengan menambah nilai K
                2. Tambah fitur yang lebih informatif
                3. Kurangi regularisasi jika ada
                4. Pertimbangkan penggunaan model yang lebih kompleks
            """)
        else:
            st.markdown("""
            * **Model menunjukkan fitting yang baik:**
                1. Performa training dan validasi seimbang
                2. Gap yang wajar antara training dan validation score
                3. Tren validation score stabil
                4. Lanjutkan dengan parameter model saat ini
            """)

        st.markdown("""
        ### Kesimpulan Learning Curve:
        
        1. **Convergence (Konvergensi)**:
           - Jika kedua garis mendatar: Model telah konvergen
           - Jika masih naik: Mungkin perlu lebih banyak data training
        
        2. **Variance (Variasi)**:
           - Area bayangan sempit: Model stabil
           - Area bayangan lebar: Model sensitif terhadap data training
        
        3. **Bias vs Variance Trade-off**:
           - High bias (underfitting): Kedua score rendah
           - High variance (overfitting): Gap besar antara training dan validation
           - Good balance: Gap kecil dengan score yang dapat diterima
        
        ### Implikasi untuk Pengembangan Model:
        
        1. **Kebutuhan Data**:
           - Jika kurva masih naik: Pertimbangkan menambah data training
           - Jika sudah mendatar: Jumlah data sudah mencukupi
        
        2. **Parameter Model**:
           - Jika overfitting: Pertimbangkan parameter yang lebih sederhana
           - Jika underfitting: Coba parameter yang lebih kompleks
        
        3. **Validasi Silang**:
           - Stabilitas cross-validation menunjukkan keandalan model
           - Variasi tinggi menunjukkan perlu penyesuaian parameter
        """)
    
    # 7. Kesimpulan
    st.header("7. Kesimpulan")
    
    # Hitung metrik-metrik untuk kesimpulan
    test_accuracy = accuracy_score(y_test, y_pred)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Analisis performa model
    def get_performance_status(accuracy):
        if accuracy >= 0.9:
            return "sangat baik"
        elif accuracy >= 0.8:
            return "baik"
        elif accuracy >= 0.7:
            return "cukup baik"
        else:
            return "perlu peningkatan"
            
    # Analisis stabilitas model
    def get_stability_status(std):
        if std < 0.02:
            return "sangat stabil"
        elif std < 0.05:
            return "stabil"
        elif std < 0.1:
            return "cukup stabil"
        else:
            return "kurang stabil"
    
    # Analisis gap antara training dan testing
    train_accuracy = accuracy_score(y_train, y_pred_train)
    accuracy_gap = abs(train_accuracy - test_accuracy)
    
    def get_gap_status(gap):
        if gap < 0.03:
            return "sangat baik (model seimbang)"
        elif gap < 0.05:
            return "baik"
        elif gap < 0.1:
            return "cukup baik tetapi perlu diperhatikan"
        else:
            return "terlalu besar (kemungkinan overfitting)"
            
    # Generate kesimpulan dinamis
    performance_status = get_performance_status(test_accuracy)
    stability_status = get_stability_status(cv_std)
    gap_status = get_gap_status(accuracy_gap)
    
    st.write("""
    Berdasarkan hasil evaluasi yang telah dilakukan, berikut kesimpulan yang dapat diambil:
    
    1. **Performa Model**:
       - Model KNN dengan k={} menunjukkan performa yang {}
       - Accuracy pada data testing: {:.2%}
       - Accuracy pada data training: {:.2%}
       
    2. **Stabilitas Model**:
       - Cross-validation menunjukkan model {}
       - Rata-rata accuracy CV: {:.2%} (±{:.2%})
       
    3. **Generalisasi Model**:
       - Gap antara training dan testing: {:.2%} ({})
       - Learning curve menunjukkan {}
       
    4. **Rekomendasi**:
    """.format(
        n_neighbors,
        performance_status,
        test_accuracy,
        train_accuracy,
        stability_status,
        cv_mean,
        cv_std * 2,
        accuracy_gap,
        gap_status,
        fitting_status
    ))
    
    # Tambahkan rekomendasi berdasarkan analisis
    if accuracy_gap > 0.1:
        st.write("""
        - Pertimbangkan untuk mengurangi kompleksitas model
        - Tambahkan lebih banyak data training jika memungkinkan
        - Lakukan feature selection untuk mengurangi noise
        """)
    elif test_accuracy < 0.7:
        st.write("""
        - Coba tingkatkan nilai k
        - Tambahkan fitur yang lebih relevan
        - Pertimbangkan penggunaan algoritma lain
        """)
    elif cv_std > 0.1:
        st.write("""
        - Lakukan feature engineering
        - Seimbangkan dataset jika terjadi imbalance
        - Coba teknik cross-validation yang berbeda
        """)
    else:
        st.write("""
        - Model sudah cukup baik dan stabil
        - Dapat dilanjutkan ke tahap deployment
        - Monitor performa secara berkala
        """)
    
    # Tambahkan visualisasi perbandingan metrik
    st.subheader("Visualisasi Perbandingan Metrik")
    
    metrics_comparison = pd.DataFrame({
        'Metric': ['Training Accuracy', 'Testing Accuracy', 'Cross-val Mean'],
        'Value': [train_accuracy, test_accuracy, cv_mean]
    })
    
    fig = plt.figure(figsize=(10, 6))
    plt.bar(metrics_comparison['Metric'], metrics_comparison['Value'])
    plt.title('Perbandingan Metrik Performa Model')
    plt.ylabel('Accuracy')
    plt.axhline(y=0.7, color='r', linestyle='--', label='Baseline (70%)')
    plt.legend()
    st.pyplot(fig)
    
    # Tambahkan interpretasi hasil
    st.markdown("### Interpretasi Hasil:")
    st.write(f"""
    - Model menunjukkan performa {performance_status} dengan accuracy testing {test_accuracy:.2%}
    - Stabilitas model {stability_status} berdasarkan standar deviasi cross-validation
    - Gap antara training dan testing {gap_status}
    """)
    
    if accuracy_gap <= 0.1 and test_accuracy >= 0.7 and cv_std <= 0.1:
        st.success("✅ Model sudah siap untuk digunakan")
    else:
        st.warning("⚠️ Model masih memerlukan penyesuaian berdasarkan rekomendasi di atas")
    
    # 8. Model Export
    st.header("8. Export Model")
    st.write("Model telah disimpan sebagai 'knn_model.pkl' dan scaler sebagai 'scaler.pkl'")
    st.write("File-file ini akan digunakan untuk melakukan prediksi pada data baru.")

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