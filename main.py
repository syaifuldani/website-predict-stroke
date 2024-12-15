import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE    
from collections import Counter

# Function for k-fold cross validation
def perform_kfold_validation(X, y, models, k=5):
    cv_results = {}
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    for name, model in models.items():
        accuracy_scores = cross_val_score(model, X, y, scoring='accuracy', cv=skf)
        precision_scores = cross_val_score(model, X, y, scoring='precision', cv=skf)
        recall_scores = cross_val_score(model, X, y, scoring='recall', cv=skf)
        f1_scores = cross_val_score(model, X, y, scoring='f1', cv=skf)
        
        cv_results[name] = {
            'Accuracy': {
                'scores': accuracy_scores,
                'mean': accuracy_scores.mean(),
                'std': accuracy_scores.std()
            },
            'Precision': {
                'scores': precision_scores,
                'mean': precision_scores.mean(), 
                'std': precision_scores.std()
            },
            'Recall': {
                'scores': recall_scores,
                'mean': recall_scores.mean(),
                'std': recall_scores.std()
            },
            'F1': {
                'scores': f1_scores,
                'mean': f1_scores.mean(),
                'std': f1_scores.std()
            }
        }
    
    return cv_results

# Set page config
st.set_page_config(page_title="Stroke Prediction Analysis", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict"])

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

    # Tambahkan fungsi ini di bagian atas, setelah import
def perform_kfold_validation(X, y, models, k=5):
    """
    Perform k-fold cross validation for multiple models
    """
    cv_results = {}
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    for name, model in models.items():
        accuracy_scores = cross_val_score(model, X, y, scoring='accuracy', cv=skf)
        precision_scores = cross_val_score(model, X, y, scoring='precision', cv=skf)
        recall_scores = cross_val_score(model, X, y, scoring='recall', cv=skf)
        f1_scores = cross_val_score(model, X, y, scoring='f1', cv=skf)
        
        cv_results[name] = {
            'Accuracy': {
                'scores': accuracy_scores,
                'mean': accuracy_scores.mean(),
                'std': accuracy_scores.std()
            },
            'Precision': {
                'scores': precision_scores,
                'mean': precision_scores.mean(), 
                'std': precision_scores.std()
            },
            'Recall': {
                'scores': recall_scores,
                'mean': recall_scores.mean(),
                'std': recall_scores.std()
            },
            'F1': {
                'scores': f1_scores,
                'mean': f1_scores.mean(),
                'std': f1_scores.std()
            }
        }
    
    return cv_results

if page == "Home":
    st.title("Klasifikasi Penyakit Stroke Menggunakan Metode KNN, Naive Bayes dan SVM")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your healthcare dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Raw Data", "Preprocessing", "Split Data", "Classification", "Evaluation"])
        
        # Load data
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df.copy()
        
        # Tab 1: Raw Data
        with tab1:
            st.header("Raw Data")
            st.write("Dataset Shape:", df.shape)
            st.write("Dataset Preview:")
            st.write(df.head())
            st.write("Dataset Information:")
            buffer = StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
            
        # Tab 2: Preprocessing
        with tab2:
            st.header("Data Preprocessing")
            
            # Drop ID column
            df.drop(columns='id', inplace=True)
            st.write("1. Removed 'id' column")
            
            # Handle missing values
            st.write("2. Handling Missing Values")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Sebelum menggunakan nilai median:")
                st.write(df.isnull().sum())
            
            df.fillna(df.median(numeric_only=True), inplace=True)
            
            with col2:
                st.write("Setelah menggunakan nilai median:")
                st.write(df.isnull().sum())
            
            # Label encoding
            st.write("3. Label Encoding for Categorical Variables")
            label_encoder = LabelEncoder()
            categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
            
            # Tampilkan data sebelum encoding
            st.write("Data sebelum Label Encoding:")
            st.write(df[categorical_columns].head())
            
            # Lakukan label encoding
            for col in categorical_columns:
                df[col] = label_encoder.fit_transform(df[col])
                st.write(f"Encoded {col}")
            
            st.write("Data setelah Label Encoding:")
            st.write(df[categorical_columns].head())
            
            st.session_state.label_encoder = label_encoder
            
            # SMOTE Balancing
            st.write("4. SMOTE Class Balancing")
            
            X = df.drop("stroke", axis=1)
            y = df["stroke"]
            
            # Tampilkan distribusi kelas sebelum SMOTE
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Distribusi kelas sebelum SMOTE:")
                st.write(pd.Series(y).value_counts())
                # Visualisasi
                fig, ax = plt.subplots()
                sns.countplot(data=pd.DataFrame(y), x='stroke')
                plt.title("Distribusi Kelas Sebelum SMOTE")
                st.pyplot(fig)
            
            # Terapkan SMOTE
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            with col2:
                st.write("Distribusi kelas setelah SMOTE:")
                st.write(pd.Series(y_balanced).value_counts())
                # Visualisasi
                fig, ax = plt.subplots()
                sns.countplot(data=pd.DataFrame(y_balanced), x='stroke')
                plt.title("Distribusi Kelas Setelah SMOTE")
                st.pyplot(fig)
            
            # Tampilkan ukuran dataset
            st.write(f"Ukuran dataset sebelum SMOTE: {X.shape}")
            st.write(f"Ukuran dataset setelah SMOTE: {X_balanced.shape}")
        
        # Tab 3: Split Data
        with tab3:
            st.header("Data Splitting and Scaling")
            
            # Prepare features and target
            X = df.drop("stroke", axis=1)
            y = df["stroke"]
            
            # Normalize data
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            st.session_state.scaler = scaler
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # SMOTE application
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            st.write("Original vs Balanced Class Distribution:")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Original:", Counter(y_train))
            with col2:
                st.write("After SMOTE:", Counter(y_train_balanced))
            
            # Display split information
            st.write(f"Training set size: {X_train.shape}")
            st.write(f"Testing set size: {X_test.shape}")
        
        # Di Tab 4 (Classification), setelah bagian model training, tambahkan:
        with tab4:
            st.header("Model Training and Cross Validation")
            
            # Model initialization
            models = {
                'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean'),
                'SVM': SVC(probability=True, kernel='rbf', C=1.0, class_weight='balanced',tol=0.10, max_iter=100),
                # 'SVM': SVC(
                #             probability=True, 
                #             kernel='rbf', 
                #             C=1.0,  # Parameter regularisasi
                #             class_weight='balanced',
                #             max_iter=100,  # Maksimum iterasi
                #             tol=0.10,  # Toleransi untuk kriteria penghentian
                #             cache_size=200,
                #             random_state=42
                #         ),
                'Naive Bayes': GaussianNB()
            }
            
            # Model training and basic evaluation
            results = {}
            trained_models = {}
            
            for name, model in models.items():
                with st.spinner(f'Training {name} model...'):
                    model.fit(X_train_balanced, y_train_balanced)
                    trained_models[name] = model
                    y_pred = model.predict(X_test)
                    
                    results[name] = {
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred),
                        'Recall': recall_score(y_test, y_pred),
                        'F1-score': f1_score(y_test, y_pred)
                    }
                    
                    st.success(f"{name} model trained successfully!")
            
            # K-fold Cross Validation
            st.subheader("K-Fold Cross Validation")
            k_folds_options = [4, 8, 12]
            k_folds = st.selectbox("Pilih jumlah K-Fold:", k_folds_options)
            
            # Perform k-fold validation
            with st.spinner("Performing cross validation..."):
                cv_results = perform_kfold_validation(X_train_balanced, 
                                                    y_train_balanced,
                                                    trained_models, 
                                                    k=k_folds)
            
            # Store results in session state
            st.session_state.models = trained_models
            st.session_state.cv_results = cv_results
        
        # Tab 5: Evaluation
        with tab5:
            st.header("Model Evaluation")
            
            # Create columns for confusion matrices
            col1, col2, col3 = st.columns(3)
            columns = [col1, col2, col3]

                    # Cross Validation Results
            st.header("Cross Validation Results")
            
            # Display results untuk k-fold yang dipilih
            for name, metrics in cv_results.items():
                st.subheader(f"{name} Cross Validation Results (K={k_folds})")
                
                # Create DataFrame for this model's results
                cv_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
                    'Mean Score': [metrics[m]['mean'] for m in ['Accuracy','Precision','Recall','F1']],
                    'Std Dev': [metrics[m]['std'] for m in ['Accuracy','Precision','Recall','F1']]
                })
                
                # Display summary table
                st.dataframe(cv_df.style.format({
                    'Mean Score': '{:.3f}',
                    'Std Dev': '{:.3f}'
                }))
                
                # Plot fold scores
                fig, ax = plt.subplots(figsize=(10, 6))
                width = 0.2
                x = np.arange(k_folds)
                
                for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1']):
                    scores = metrics[metric]['scores']
                    ax.bar(x + i*width, scores, width, label=metric)
                
                ax.set_title(f'{name} - Scores per Fold (K={k_folds})')
                ax.set_xlabel('Fold')
                ax.set_ylabel('Score')
                ax.set_xticks(x + width*1.5)
                ax.set_xticklabels([f'Fold {i+1}' for i in range(k_folds)])
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                    
            # Plot confusion matrices side by side
            for (name, model), col in zip(trained_models.items(), columns):
                with col:
                    st.subheader(f"{name}")
                    y_pred = model.predict(X_test)
                    
                    # Confusion Matrix
                    fig, ax = plt.subplots(figsize=(6, 5))
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["No Stroke", "Stroke"],
                            yticklabels=["No Stroke", "Stroke"], ax=ax)
                    plt.xlabel("Predicted")
                    plt.ylabel("True")
                    plt.title(f"Confusion Matrix")
                    st.pyplot(fig)
                    plt.close()
                    
                    # Classification Report dalam format tabel
                    st.write("Classification Report:")
                    
                    # Buat dataframe untuk report
                    precision = precision_score(y_test, y_pred, average=None)
                    recall = recall_score(y_test, y_pred, average=None)
                    f1 = f1_score(y_test, y_pred, average=None)
                    support = np.bincount(y_test.astype(int))
                    
                    # Buat DataFrame untuk metrik per kelas
                    df_report = pd.DataFrame({
                        'precision': precision,
                        'recall': recall,
                        'f1-score': f1,
                        'support': support
                    }, index=['No Stroke', 'Stroke'])
                    
                    # Tambahkan accuracy
                    accuracy = accuracy_score(y_test, y_pred)
                    df_report.loc['accuracy'] = [accuracy, accuracy, accuracy, sum(support)]
                    
                    # Tambahkan macro avg
                    macro_p = np.mean(precision)
                    macro_r = np.mean(recall)
                    macro_f1 = np.mean(f1)
                    df_report.loc['macro avg'] = [macro_p, macro_r, macro_f1, sum(support)]
                    
                    # Tambahkan weighted avg
                    weighted_p = precision_score(y_test, y_pred, average='weighted')
                    weighted_r = recall_score(y_test, y_pred, average='weighted')
                    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
                    df_report.loc['weighted avg'] = [weighted_p, weighted_r, weighted_f1, sum(support)]
                    
                    # Format angka dalam DataFrame
                    df_report = df_report.round(3)
                    
                    # Tampilkan sebagai tabel yang rapi
                    st.dataframe(df_report.style.set_properties(**{
                        'text-align': 'center',
                        'font-size': '14px',
                        'padding': '5px'
                    }))
            
            # Model Comparison
            st.subheader("Model Performance Comparison")
            metrics_df = pd.DataFrame(results).T
            
            # Bar plot dan Heatmap dalam 2 kolom
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar plot dengan format persentase
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Konversi nilai ke persentase (dikalikan 100)
                metrics_df_percent = metrics_df * 100
                
                # Plot bar chart
                metrics_df_percent.plot(kind='bar', ax=ax)
                
                # Konfigurasi tampilan
                plt.title('Komparasi Performa')
                plt.xlabel('Models')
                plt.ylabel('Score (%)')
                plt.legend(bbox_to_anchor=(1.05, 1))
                
                # Tambahkan label persentase di atas bar
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.1f%%', padding=3)
                
                # Format y-axis dalam persen
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                        
            with col2:
                # Heatmap
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(metrics_df, annot=True, cmap='YlOrRd', fmt='.3f', ax=ax)
                plt.title('Performance Metrics Heatmap')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Best models
            st.subheader("Best Models per Metric")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                for metric in metrics_df.columns:
                    best_model = metrics_df[metric].idxmax()
                    best_score = metrics_df[metric].max()
                    st.write(f"**{metric}:** {best_model} ({best_score:.3f})")

elif page == "Predict":
    st.title("Stroke Prediction")
    
    if st.session_state.models is None:
        st.warning("Please train the models in the Home page first!")
    else:
        st.write("Enter patient information for prediction:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=0, max_value=120, value=50)
            hypertension = st.selectbox("Hypertension", [0, 1])
            heart_disease = st.selectbox("Heart Disease", [0, 1])
            ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        
        with col2:
            work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Never_worked", "children"])
            residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
            avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=90.0)
            bmi = st.number_input("BMI", min_value=0.0, value=25.0)
            smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

        if st.button("Predict"):
            # Prepare input data
            input_data = pd.DataFrame({
                'gender': [gender],
                'age': [age],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'ever_married': [ever_married],
                'work_type': [work_type],
                'Residence_type': [residence_type],
                'avg_glucose_level': [avg_glucose_level],
                'bmi': [bmi],
                'smoking_status': [smoking_status]
            })
            
            # Apply the same preprocessing
            categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
            for col in categorical_columns:
                input_data[col] = st.session_state.label_encoder.fit_transform(input_data[col])
            
            # Scale the input data
            input_scaled = st.session_state.scaler.transform(input_data)
            
            # Make predictions with all models
            st.subheader("Predictions:")
            for name, model in st.session_state.models.items():
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0]
                
                st.write(f"{name} Model:")
                if prediction == 0:
                    st.write(f"Prediction: No Stroke (Probability: {probability[0]:.2%})")
                else:
                    st.write(f"Prediction: Stroke (Probability: {probability[1]:.2%})")
                st.write("---")

else:
    st.warning("Please select a valid page from the sidebar.")