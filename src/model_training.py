import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Path setup
DATA_PATH = os.path.join('data', 'seeds_dataset.csv')
CONF_MATRIX_PATH = os.path.join('visualizations', 'confusion_matrix.png')
FEAT_IMP_PATH = os.path.join('visualizations', 'feature_importance.png')
CLASS_DIST_PATH = os.path.join('visualizations', 'class_distribution.png')
MODEL_PATH = os.path.join('src', 'rf_model.joblib')

# Ensure visualizations directory exists
os.makedirs('visualizations', exist_ok=True)

# 1. Load dataset
# (Use the correct path if running from root)
df = pd.read_csv(DATA_PATH)

# 2. EDA
print("5 Baris Pertama:")
print(df.head())

print("\nInfo Data:")
print(df.info())

print("\nStatistik Deskriptif:")
print(df.describe())

# 3. Cek distribusi kelas
print("\nDistribusi Kelas:")
print(df['target'].value_counts())
# Mapping label ke nama varietas
label_map = {0: 'Kama', 1: 'Rosa', 2: 'Canadian'}
df['varietas'] = df['target'].map(label_map)
plt.figure(figsize=(6,4))
sns.countplot(x='varietas', data=df, palette='pastel')
plt.title('Distribusi Kelas')
plt.xlabel('Varietas')
plt.ylabel('Jumlah')
plt.tight_layout()
plt.savefig(CLASS_DIST_PATH)
plt.close()

# 4. Normalisasi data
X = df.drop(['target', 'varietas'], axis=1)
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 6. Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 7. Evaluasi
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Akurasi: {acc:.4f}')
print('\nLaporan Klasifikasi:')
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Kama', 'Rosa', 'Canadian'],
            yticklabels=['Kama', 'Rosa', 'Canadian'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH)
plt.close()

# Feature importance
importances = rf.feature_importances_
features = X.columns
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features, palette='viridis')
plt.title('Feature Importance pada Random Forest')
plt.xlabel('Importance')
plt.ylabel('Fitur')
plt.tight_layout()
plt.savefig(FEAT_IMP_PATH)
plt.close()

# 8. Save model
joblib.dump((rf, scaler), MODEL_PATH)
print(f'Model saved to {MODEL_PATH}')
