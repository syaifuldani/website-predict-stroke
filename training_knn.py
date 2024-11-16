import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Tampilkan distribusi awal gender
print("\nDistribusi Gender sebelum preprocessing:")
print(df['gender'].value_counts())

# Hapus baris dengan gender 'Other'
df = df[df['gender'] != 'Other']

print("\nDistribusi Gender setelah menghapus kategori 'Other':")
print(df['gender'].value_counts())

# Drop id column
df = df.drop('id', axis=1)

# Remove rows with 'Unknown' smoking status
df = df[df['smoking_status'] != 'Unknown']

# Handle missing values in bmi column using median
bmi_median = df['bmi'].median()
df['bmi'] = df['bmi'].fillna(bmi_median)

print("\nDistribusi data setelah preprocessing awal:")
print(f"Total data: {len(df)}")
print("\nDistribusi Stroke:")
print(df['stroke'].value_counts(normalize=True) * 100)

# Use LabelEncoder for categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])
    print(f"\nEncoding untuk {column}:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"{label}: {i}")

# Split features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Scale numerical features
scaler = MinMaxScaler()
numerical_cols = ['age', 'avg_glucose_level', 'bmi']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Try different values of k using GridSearchCV
param_grid = {'n_neighbors': range(1, 21, 2)}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Use best model
best_model = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Tampilkan perubahan dalam dataset
print("\nInformasi Dataset:")
print(f"Jumlah data awal: {len(df)}")
print(f"Jumlah fitur: {len(df.columns)}")
print("\nDistribusi kelas setelah preprocessing:")
print(y.value_counts(normalize=True) * 100)

# Save model and preprocessing objects
pickle.dump(best_model, open('knn_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Save the encoders
encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    le.fit(df[column])
    encoders[column] = le
pickle.dump(encoders, open('label_encoders.pkl', 'wb'))

print("\nModel, scaler, dan encoders telah disimpan!")