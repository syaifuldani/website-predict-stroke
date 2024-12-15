import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Definisi kategori standar
CATEGORIES = {
    'gender': ['Female', 'Male', 'Other'],
    'ever_married': ['No', 'Yes'],
    'work_type': ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children'],
    'Residence_type': ['Rural', 'Urban'],
    'smoking_status': ['formerly smoked', 'never smoked', 'smokes', 'Unknown']
}

# Load data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

print("\nDistribusi awal data:")
print("Distribusi Gender:")
print(df['gender'].value_counts())
print("\nDistribusi Stroke:")
print(df['stroke'].value_counts())

# Drop id column
df = df.drop('id', axis=1)

# Handle missing values
bmi_median = df['bmi'].median()
df['bmi'] = df['bmi'].fillna(bmi_median)

# Encoding yang konsisten
encoders = {}
for column, categories in CATEGORIES.items():
    le = LabelEncoder()
    le.fit(categories)
    df[column] = le.transform(df[column])
    encoders[column] = le
    print(f"\nEncoding untuk {column}:")
    for i, label in enumerate(le.classes_):
        print(f"{label}: {i}")

# Split features dan target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Scale fitur numerik
scaler = MinMaxScaler()
numerical_cols = ['age', 'avg_glucose_level', 'bmi']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE untuk balance data training
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nDistribusi setelah SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

# Grid search dengan data balanced
param_grid = {
    'n_neighbors': range(1, 21, 2),
    'weights': ['uniform', 'distance']
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(
    knn, 
    param_grid, 
    cv=5, 
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train_balanced, y_train_balanced)

print("\nParameter terbaik:", grid_search.best_params_)
print("Skor cross-validation terbaik:", grid_search.best_score_)

# Evaluasi model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, zero_division=1))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature importance berdasarkan korelasi
correlation = X_train.corrwith(pd.Series(y_train))
plt.figure(figsize=(10, 6))
correlation.abs().sort_values(ascending=False).plot(kind='bar')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Simpan model dan preprocessing
pickle.dump(best_model, open('knn_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(encoders, open('label_encoders.pkl', 'wb'))

print("\nModel dan komponen preprocessing telah disimpan!")

# Function untuk prediksi
def predict_stroke(input_data):
    """
    input_data: dictionary dengan format:
    {
        'gender': str,
        'age': float,
        'hypertension': int,
        'heart_disease': int,
        'ever_married': str,
        'work_type': str,
        'Residence_type': str,
        'avg_glucose_level': float,
        'bmi': float,
        'smoking_status': str
    }
    """
    # Load model dan preprocessing
    model = pickle.load(open('knn_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    encoders = pickle.load(open('label_encoders.pkl', 'rb'))
    
    # Convert input ke DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical features
    for column in CATEGORIES.keys():
        input_df[column] = encoders[column].transform(input_df[column])
    
    # Scale numerical features
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Predict
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    return prediction[0], probability[0]

# Contoh penggunaan
sample_input = {
    'gender': 'Male',
    'age': 67,
    'hypertension': 0,
    'heart_disease': 1,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 228.69,
    'bmi': 36.6,
    'smoking_status': 'formerly smoked'
}

prediction, probability = predict_stroke(sample_input)
print("\nContoh Prediksi:")
print(f"Prediksi: {'Stroke' if prediction == 1 else 'Tidak Stroke'}")
print(f"Probabilitas: {probability}")