# Random Forest Seeds Classification

Proyek ini menerapkan algoritma Random Forest untuk klasifikasi varietas biji gandum berdasarkan karakteristik morfologis. Terdapat aplikasi web berbasis Flask untuk prediksi interaktif.

## Struktur Proyek
- `data/seeds_dataset.csv`: Dataset biji gandum
- `src/model_training.py`: Script pelatihan & evaluasi model
- `aplikasi/`: Web app Flask
- `visualizations/`: Visualisasi confusion matrix & feature importance

## Cara Menjalankan
1. Install dependencies: `pip install -r requirements.txt`
2. Jalankan training: `python src/model_training.py`
3. Jalankan web: `python aplikasi/app.py`

## Deployment
- Gunakan Railway, dengan `Procfile` yang sudah tersedia.

## Dataset
Sumber: UCI ML Repository (Seeds Dataset)
