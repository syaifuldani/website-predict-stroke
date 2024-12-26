import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import io

# Set page config
st.set_page_config(
    page_title="Prediksi Stroke",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'preprocessed_data' not in st.session_state:
    st.session_state['preprocessed_data'] = None
if 'target' not in st.session_state:
    st.session_state['target'] = None

# Sidebar
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Data Upload & Preview", "Preprocessing", "Modeling", "Prediction"])

# Title
st.title("Prediksi Stroke Metode KNN, Naive Bayes, dan SVM")

# Dataset Information and Download Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.header("Dataset Information")
st.sidebar.markdown("""

**Source**: [Stroke Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

You can download the dataset directly from Kaggle using the link below:
[Download Dataset from Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/download)

*Note: You'll need a Kaggle account to download the dataset.*
""")

# File upload - always visible in sidebar
st.sidebar.markdown("---")
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    if st.session_state['data'] is None:
        st.session_state['data'] = pd.read_csv(uploaded_file)

if menu == "Data Upload & Preview":
    if st.session_state['data'] is not None:
        st.header("Data Preview")
        
        # Basic Data Overview
        with st.expander("🔍 Raw Data Sample", expanded=True):
            st.dataframe(st.session_state['data'].head())
        
        # Statistical Description
        with st.expander("📊 Statistical Description"):
            st.write(st.session_state['data'].describe())
        
        # Dataset Information
        with st.expander("ℹ️ Dataset Information"):
            buffer = io.StringIO()
            st.session_state['data'].info(buf=buffer)
            st.text(buffer.getvalue())
        
        # Missing Values Analysis
        with st.expander("❓ Missing Values Analysis"):
            missing_values = st.session_state['data'].isnull().sum()
            st.write(missing_values[missing_values > 0])
        
        # Data Distribution
        with st.expander("📈 Data Distribution Visualizations"):
            # Numeric columns distribution
            numeric_cols = st.session_state['data'].select_dtypes(include=['float64', 'int64']).columns
            selected_num_col = st.selectbox("Select numeric column for distribution plot", numeric_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=st.session_state['data'], x=selected_num_col, ax=ax)
            st.pyplot(fig)
            
            # Categorical columns distribution
            categorical_cols = st.session_state['data'].select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                selected_cat_col = st.selectbox("Select categorical column for count plot", categorical_cols)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(data=st.session_state['data'], x=selected_cat_col, ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

elif menu == "Preprocessing":
    if st.session_state['data'] is not None:
        st.header("Tahapan Preprocessing")
        
        # Initialize preprocessing state if not exists
        if 'preprocessing_done' not in st.session_state:
            st.session_state['preprocessing_done'] = {
                'clean': False,
                'encode': False,
                'scale': False
            }
            st.session_state['current_data'] = st.session_state['data'].copy()
        
        with st.expander("1️⃣ Data Cleaning", expanded=True):
            clean_tab1, clean_tab2 = st.tabs(["Penghapusan Fitur", "Handle Missing Values"])
            
            # Untuk bagian Data Cleaning di Preprocessing
            with clean_tab1:
                st.markdown("### Menghapus Fitur yang kurang relevan")
                col1, col2 = st.columns(2)
                
                # Store original data before any changes
                if 'original_data' not in st.session_state:
                    st.session_state['original_data'] = st.session_state['data'].copy()
                
                with col1:
                    st.write("Sebelum dilakukan penghapusan:")
                    st.dataframe(st.session_state['original_data'].head())
                
                if not st.session_state['preprocessing_done']['clean']:
                    # Periksa apakah kolom 'id' ada sebelum menghapusnya
                    columns_to_drop = []
                    if 'id' in st.session_state['current_data'].columns:
                        columns_to_drop.append('id')
                    
                    if columns_to_drop:
                        st.session_state['current_data'] = st.session_state['current_data'].drop(columns_to_drop, axis=1)
                    else:
                        st.info("No unnecessary features to remove")
                
                with col2:
                    st.write("Setelah dilakukan penghapusan:")
                    st.dataframe(st.session_state['current_data'].head())
            
            with clean_tab2:
                st.markdown("### 1.2 Handle Missing Values")
                col3, col4 = st.columns(2)
                
                # Store original missing values if not already stored
                if 'original_missing_values' not in st.session_state:
                    st.session_state['original_missing_values'] = st.session_state['current_data'].isnull().sum()
                    st.session_state['original_missing_df'] = st.session_state['current_data'].copy()
                
                with col3:
                    st.write("Missing values sebelum dihandle:")
                    missing_before = st.session_state['original_missing_values']
                    if any(missing_before > 0):
                        st.write(missing_before[missing_before > 0])
                        st.write("\nSampel data dengan missing values:")
                        # Tampilkan beberapa baris yang memiliki missing values
                        missing_rows = st.session_state['original_missing_df'][
                            st.session_state['original_missing_df'].isnull().any(axis=1)
                        ].head()
                        st.dataframe(missing_rows)
                    else:
                        st.success("Tidak ada missing values dalam dataset")
                
                # Handle missing values only if not done before
                if not st.session_state['preprocessing_done']['clean']:
                    # Simpan nilai median sebelum mengisi missing values
                    if 'median_values' not in st.session_state:
                        st.session_state['median_values'] = st.session_state['current_data'].median(numeric_only=True)
                    
                    # Isi missing values dengan median
                    st.session_state['current_data'] = st.session_state['current_data'].fillna(
                        st.session_state['median_values']
                    )
                    st.session_state['preprocessing_done']['clean'] = True
                
                with col4:
                    st.write("Missing values setelah dihandle:")
                    missing_after = st.session_state['current_data'].isnull().sum()
                    if missing_after.sum() == 0:
                        st.success("Semua missing values telah dihandle")
                        st.write("\nSampel data setelah handling:")
                        # Tampilkan data yang sama setelah handling
                        if 'original_missing_df' in st.session_state:
                            handled_rows = st.session_state['current_data'].iloc[
                                st.session_state['original_missing_df'].isnull().any(axis=1).index
                            ].head()
                            st.dataframe(handled_rows)
                    else:
                        st.write(missing_after[missing_after > 0])
        
        # Feature Engineering Expander
        with st.expander("2️⃣ Feature Engineering"):
            feature_tab1, feature_tab2 = st.tabs(["Label Encoding Variabel Kategori", "Binary Conversion"])
            
            with feature_tab1:
                st.markdown("### 2.1 Categorical Variable Encoding")
                col5, col6 = st.columns(2)
                categorical_cols = st.session_state['current_data'].select_dtypes(include=['object']).columns
                
                # Store original categorical data before encoding
                if not st.session_state['preprocessing_done']['encode']:
                    if 'original_categorical_data' not in st.session_state:
                        st.session_state['original_categorical_data'] = st.session_state['current_data'][categorical_cols].copy() if len(categorical_cols) > 0 else None
                
                with col5:
                    st.write("Data sebelum encoding:")
                    if st.session_state.get('original_categorical_data') is not None:
                        st.write("Categorical columns:", list(st.session_state['original_categorical_data'].columns))
                        st.dataframe(st.session_state['original_categorical_data'].head())
                    else:
                        st.info("No categorical columns found in the dataset")
                
                if not st.session_state['preprocessing_done']['encode'] and len(categorical_cols) > 0:
                    le = LabelEncoder()
                    for col in categorical_cols:
                        st.session_state['current_data'][col] = le.fit_transform(st.session_state['current_data'][col])
                    st.session_state['preprocessing_done']['encode'] = True
                
                with col6:
                    st.write("Data setelah encoding:")
                    if st.session_state.get('original_categorical_data') is not None:
                        # Display the encoded version of the same categorical columns
                        encoded_cats = st.session_state['current_data'][st.session_state['original_categorical_data'].columns]
                        st.dataframe(encoded_cats.head())
                    else:
                        st.info("No categorical columns to encode")
                
                if st.session_state.get('original_categorical_data') is not None:
                    st.info("""Pengkodean mengonversi nilai kategorikal menjadi numerik:
- Gender: Female=0, Male=1
- Ever Married: No=0, Yes=1
- Work Type: Children=0, Govt job=1, Never worked=2, Private=3, Self-employed=4
- Residence Type: Rural=0, Urban=1
- Smoking Status: Unknown=0, formerly smoked=1, never smoked=2, smokes=3""")
                    
        with feature_tab2:
            st.markdown("### 2.2 Binary Variable Conversion")
            col7, col8 = st.columns(2)
            binary_cols = ['hypertension', 'heart_disease']
            
            with col7:
                st.write("Binary data:")
                st.dataframe(st.session_state['current_data'][binary_cols].head())
            
            with col8:
                st.write("Binary data format (0/1):")
                st.markdown("""
                - 0: No
                - 1: Yes
                """)
                
        # Step 3: Feature Scaling
        with st.expander("3️⃣ Feature Scaling"):
            scale_tab1, = st.tabs(["Normalisasi Min-Max"])
        
        with scale_tab1:
            st.markdown("### 3.1 Normalisasi Min-Max scaller")
            col9, col10 = st.columns(2)
            numerical_cols = ['age', 'avg_glucose_level', 'bmi']
            
            with col9:
                st.write("Data numerik sebelum normalisasi:")
                st.dataframe(st.session_state['current_data'][numerical_cols].head())
            
            if not st.session_state['preprocessing_done']['scale']:
                scaler = MinMaxScaler()
                st.session_state['current_data'][numerical_cols] = scaler.fit_transform(
                    st.session_state['current_data'][numerical_cols]
                )
                st.session_state['preprocessing_done']['scale'] = True
            
            with col10:
                st.write("Data numerik setelah normalisasi (rentang 0-1):")
                st.dataframe(st.session_state['current_data'][numerical_cols].head())
        
        # Tambahkan Step 4: Data Splitting (sebelum SMOTE)
        with st.expander("4️⃣ Splitting Data"):
            st.markdown("### 4.1 Membagi Data Training dan Testing")
            
            if 'train_test_data' not in st.session_state:
                X = st.session_state['current_data'].drop('stroke', axis=1)
                y = st.session_state['current_data']['stroke']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                st.session_state['train_test_data'] = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test
                }
            
            col_split1, col_split2 = st.columns(2)
            
            with col_split1:
                st.write("Training Data (80%):")
                st.write(f"X_train shape: {st.session_state['train_test_data']['X_train'].shape}")
                st.write(f"y_train shape: {st.session_state['train_test_data']['y_train'].shape}")
                
                # Tampilkan distribusi kelas di training set
                train_dist = pd.Series(st.session_state['train_test_data']['y_train']).value_counts().sort_index()
                st.write("\nDistribusi kelas pada data training:")
                st.write(pd.DataFrame({
                    'Kelas': ['No Stroke (0)', 'Stroke (1)'],
                    'Jumlah': train_dist.values
                }))
            
            with col_split2:
                st.write("Testing Data (20%):")
                st.write(f"X_test shape: {st.session_state['train_test_data']['X_test'].shape}")
                st.write(f"y_test shape: {st.session_state['train_test_data']['y_test'].shape}")
                
                # Tampilkan distribusi kelas di testing set
                test_dist = pd.Series(st.session_state['train_test_data']['y_test']).value_counts().sort_index()
                st.write("\nDistribusi kelas pada data testing:")
                st.write(pd.DataFrame({
                    'Kelas': ['No Stroke (0)', 'Stroke (1)'],
                    'Jumlah': test_dist.values
                }))


        # Di bagian SMOTE expander
        with st.expander("5️⃣ Balancing Data dengan SMOTE"):
            st.markdown("### 4.1 Penerapan SMOTE untuk Mengatasi Imbalance Data")
            
            # Checkbox untuk menampilkan visualisasi
            show_viz = st.checkbox("Tampilkan Visualisasi Distribusi Data", value=False)
            
            col11, col12 = st.columns(2)
            
            # Store original data distribution if not already stored
            if 'original_distribution' not in st.session_state:
                st.session_state['original_distribution'] = st.session_state['current_data']['stroke'].value_counts().sort_index()
            
            with col11:
                st.write("Distribusi kelas sebelum SMOTE:")
                if show_viz:
                    fig1, ax1 = plt.subplots(figsize=(8, 6))
                    plt.bar(["No Stroke (0)", "Stroke (1)"], st.session_state['original_distribution'].values, 
                        color=['steelblue', 'coral'])
                    plt.title("Distribusi Kelas Original")
                    plt.xlabel("Kelas")
                    plt.ylabel("Jumlah")
                    plt.ylim(0, max(st.session_state['original_distribution'].values) * 1.1)
                    st.pyplot(fig1)
                st.write("Jumlah data per kelas:")
                st.write(pd.DataFrame({
                    'Kelas': ['No Stroke (0)', 'Stroke (1)'],
                    'Jumlah': st.session_state['original_distribution'].values
                }))
            
            # Terapkan SMOTE jika belum dilakukan
            if 'smoted_data' not in st.session_state:
                X = st.session_state['current_data'].drop('stroke', axis=1)
                y = st.session_state['current_data']['stroke']
                
                smote = SMOTE(
                    sampling_strategy='auto',
                    k_neighbors=5,
                    random_state=42
                )
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
                # Gabungkan kembali data yang sudah di-resample
                resampled_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                                        pd.Series(y_resampled, name='stroke')], axis=1)
                
                st.session_state['smoted_data'] = resampled_data
            
            # Hitung distribusi setelah SMOTE
            smoted_distribution = st.session_state['smoted_data']['stroke'].value_counts().sort_index()
            
            with col12:
                st.write("Distribusi kelas setelah SMOTE:")
                if show_viz:
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    plt.bar(["No Stroke (0)", "Stroke (1)"], smoted_distribution.values,
                        color=['steelblue', 'coral'])
                    plt.title("Distribusi Kelas Setelah SMOTE")
                    plt.xlabel("Kelas")
                    plt.ylabel("Jumlah")
                    plt.ylim(0, max(smoted_distribution.values) * 1.1)
                    st.pyplot(fig2)
                st.write("Jumlah data per kelas setelah SMOTE:")
                st.write(pd.DataFrame({
                    'Kelas': ['No Stroke (0)', 'Stroke (1)'],
                    'Jumlah': smoted_distribution.values
                }))
            
            # Update current data dengan data yang sudah di-SMOTE
            if not st.session_state['preprocessing_done'].get('smote', False):
                st.session_state['current_data'] = st.session_state['smoted_data']
                st.session_state['preprocessing_done']['smote'] = True
                
        with st.expander("✅ Hasil final dari tahap PreProcessing", expanded=True):
            st.write("Hasil final dari semua tahap PreProcessing:")
            st.dataframe(st.session_state['current_data'].head())
            
            # Save processed data
            if all(st.session_state['preprocessing_done'].values()):
                X = st.session_state['current_data'].drop("stroke", axis=1)
                y = st.session_state['current_data']['stroke']
                st.session_state['preprocessed_data'] = X
                st.session_state['target'] = y
                # st.success("✅ All preprocessing steps have been applied successfully!")

elif menu == "Modeling":
    if st.session_state['preprocessed_data'] is not None and st.session_state['target'] is not None:
        st.header("Model Training and Evaluation")

        # Initialize models with optimized parameters
        models = {
            'KNN': KNeighborsClassifier(
                n_neighbors=3,
                weights='distance',
                metric='euclidean',
                algorithm='auto',
                leaf_size=30,
                p=2
            ),
            
            'SVM': SVC(
                probability=True,
                kernel='rbf',
                C=10.0,
                gamma='scale',
                class_weight='balanced',
                tol=0.001,
            ),
            'Naive Bayes': GaussianNB(
                var_smoothing=1e-8
            )
        }

        # Define k-fold variations
        k_folds = [5, 7, 12]

        overall_metrics = []  # To store metrics for all models and folds

        if st.button("Train and Evaluate Models"):
            # Split data with a fixed test size of 0.2
            X = st.session_state['preprocessed_data']
            y = st.session_state['target']

            X = X.fillna(X.median(numeric_only=True))

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            smote = SMOTE(
                sampling_strategy='auto',
                k_neighbors=5,
                random_state=42
            )
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

            for name, model in models.items():
                st.subheader(f"{name} Model Evaluation")
                model_results  = []
                for k in k_folds:
                    with st.expander(f"{k}-fold Cross Validation", expanded=False):
                        # Train model
                        model.fit(X_train_balanced, y_train_balanced)
                        y_pred = model.predict(X_test)

                        # Cross validation with current k
                        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
                        cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced,
                                                    cv=cv, scoring='accuracy')
                        
                        report = classification_report(y_test, y_pred, output_dict=True)
                        metrics = {
                            'Model': name,
                            'K-Fold': k,
                            'CV Accuracy': cv_scores.mean(),
                            'Accuracy': report['accuracy'],
                            'Precision': report['weighted avg']['precision'],
                            'Recall': report['weighted avg']['recall'],
                            'F1-score': report['weighted avg']['f1-score']
                        }
                        model_results.append(metrics)
                        overall_metrics.append(metrics)

                        # Display cross-validation results
                        st.write(f"{k}-Fold Cross-validation Results:")
                        cv_results_df = pd.DataFrame({
                            'Fold': range(1, k+1),
                            'Accuracy': cv_scores
                        })
                        st.dataframe(cv_results_df)
                        st.write(f"Mean CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std() * 2:.3f})")

                        # Classification Report
                        st.write("\nClassification Report:")
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df)

                        # Confusion Matrix
                        st.write("\nConfusion Matrix:")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        cm = confusion_matrix(y_test, y_pred)
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                   xticklabels=["No Stroke", "Stroke"],
                                   yticklabels=["No Stroke", "Stroke"])
                        plt.title(f"Confusion Matrix for {name} ({k}-fold CV)")
                        plt.xlabel("Predicted")
                        plt.ylabel("True")
                        st.pyplot(fig)

                # Compare and display results for current model across different k-folds
                comparison_df = pd.DataFrame(model_results)
                st.write(f"\nPerformance Summary for {name}:")
                st.dataframe(comparison_df.round(3))

                # Create DataFrame for results
            metrics_df = pd.DataFrame(overall_metrics)
            # pivot_df = metrics_df.pivot_table(index='Model', values=['CV Accuracy', 'Accuracy', 'Precision', 'Recall', 'F1-score'], aggfunc=np.mean)
            pivot_df = metrics_df.pivot_table(index='Model', values=['Accuracy', 'Precision', 'Recall', 'F1-score'], aggfunc=np.mean)

            # Plotting
            with st.expander("📈 Model Performance Comparison", expanded=True):
                fig, ax = plt.subplots(figsize=(12, 7))  # Adjusted figure size for better readability
                # Convert scores to percentage
                pivot_df_percentage = pivot_df * 100  # Multiply by 100 to convert to percentage
                bars = pivot_df_percentage.plot(kind='bar', ax=ax, width=0.8, legend=False)  # Adjust bar width if necessary
                plt.title('Model Performance Comparison')
                plt.ylabel('Score (%)')  # Update y-label to indicate percentage
                plt.xticks(rotation=0)
                plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)

                # Adding percentage labels above each bar
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='center', fontsize=10, color='black', rotation=0, xytext=(0, 10),
                                textcoords='offset points')

                plt.tight_layout()
                st.pyplot(fig)

                # Display DataFrame in percentage format
                st.dataframe(pivot_df_percentage.round(2))  # Round to 2 decimal places for better readability



elif menu == "Prediction":
    if st.session_state['preprocessed_data'] is not None:
        st.header("Stroke Prediction")
        
        # Define columns for preprocessing
        numerical_columns = ['age', 'avg_glucose_level', 'bmi']
        categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        binary_columns = ['hypertension', 'heart_disease']
        
        with st.expander("📋 Input Patient Information", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Personal Information")
                gender = st.selectbox("Gender", ["Male", "Female"])
                age = st.number_input("Age", min_value=0, max_value=120, value=65)  # Default ke usia berisiko
                hypertension = st.selectbox("Hypertension", ["No", "Yes"])
                heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
                ever_married = st.selectbox("Ever Married", ["No", "Yes"])
                
            with col2:
                st.subheader("Additional Information")
                work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
                residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
                avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=500.0, value=240.0)
                bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=32.0)
                smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", 'unknown'])
        
        if st.button("Make Prediction"):
            # Initialize optimized models with better parameters for stroke detection
            models = {
                'KNN': KNeighborsClassifier(
                    n_neighbors=3,
                    weights='distance',
                    metric='euclidean'
                ),
                'SVM': SVC(
                    probability=True,
                    kernel='rbf',
                    C=100.0,
                    gamma='auto',
                    class_weight='balanced',
                    random_state=42
                ),
                'Naive Bayes': GaussianNB()
            }
            
            # Prepare input data
            input_data = {
                'gender': gender,
                'age': age,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'ever_married': ever_married,
                'work_type': work_type,
                'Residence_type': residence_type,
                'avg_glucose_level': avg_glucose_level,
                'bmi': bmi,
                'smoking_status': smoking_status
            }
            
            # Preprocess input data
            df_input = pd.DataFrame([input_data])
            
            # Encode categorical variables consistently
            encoders = {
                'gender': ['Female', 'Male'],
                'ever_married': ['No', 'Yes'],
                'work_type': ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'],
                'Residence_type': ['Rural', 'Urban'],
                'smoking_status': ['never smoked', 'formerly smoked', 'smokes', 'unknown']
            }
            
            for col in categorical_columns:
                le = LabelEncoder()
                le.fit(encoders[col])
                df_input[col] = le.transform(df_input[col])
            
            # Convert binary variables
            df_input['hypertension'] = df_input['hypertension'].map({'No': 0, 'Yes': 1})
            df_input['heart_disease'] = df_input['heart_disease'].map({'No': 0, 'Yes': 1})
            
            with st.expander("🎯 Prediction Results", expanded=True):
                # Process training data
                X = st.session_state['preprocessed_data']
                y = st.session_state['target']
                
                # Handle missing values
                X = X.fillna(X.median(numeric_only=True))
                
                # Scale features
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)
                df_input_scaled = scaler.transform(df_input)
                
                # Create columns for each model
                cols = st.columns(len(models))
                
                for idx, (name, model) in enumerate(models.items()):
                    with cols[idx]:
                        st.markdown(f"### {name}")
                        
                        # Apply SMOTE
                        smote = SMOTE(random_state=42)
                        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
                        
                        # Train model
                        model.fit(X_resampled, y_resampled)
                        
                        # Make prediction
                        prediction = model.predict(df_input_scaled)
                        probability = model.predict_proba(df_input_scaled)
                        
                        # Determine risk level based on probabilities and risk factors
                        stroke_prob = probability[0][1]
                        
                        # Count risk factors
                        risk_factors = []
                        if age > 55:
                            risk_factors.append("- Usia di atas 55 tahun")
                        if hypertension == "Yes":
                            risk_factors.append("- Memiliki hipertensi")
                        if heart_disease == "Yes":
                            risk_factors.append("- Memiliki penyakit jantung")
                        if avg_glucose_level > 200:
                            risk_factors.append("- Kadar glukosa tinggi")
                        if bmi > 30:
                            risk_factors.append("- BMI menunjukkan obesitas")
                        if smoking_status == "smokes":
                            risk_factors.append("- Status perokok aktif")
                        
                        # Adjust prediction based on risk factors
                        high_risk = len(risk_factors) >= 3 or stroke_prob > 0.3
                        
                        if high_risk:
                            st.error("🚨 High Risk of Stroke")
                            message = "High risk detected based on:"
                        else:
                            st.success("✅ Low Risk of Stroke")
                            message = "Risk factors present:"
                        
                        # Display probabilities
                        st.write("\nDetail Probabilitas:")
                        st.write(f"Probabilitas Stroke: {stroke_prob:.2%}")
                        st.write(f"Probabilitas Tidak Stroke: {probability[0][0]:.2%}")
                        
                        # Display risk factors
                        if risk_factors:
                            st.write(f"\n{message}")
                            for factor in risk_factors:
                                st.write(factor)
                            
                            if len(risk_factors) >= 3:
                                st.warning("⚠️ Multiple high-risk factors detected!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit for Stroke Prediction Analysis")