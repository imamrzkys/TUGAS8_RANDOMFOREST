import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'rf_model.joblib')
model, scaler = joblib.load(MODEL_PATH)

# Feature names (in order)
FEATURES = [
    'A', 'P', 'C', 'LK', 'WK', 'A_Coef', 'LKG'
]

# Feature ranges for input filter (min, max)
FEATURE_RANGES = {
    'A': (10.59, 21.18),
    'P': (12.41, 17.25),
    'C': (0.8081, 0.9183),
    'LK': (4.899, 6.666),
    'WK': (2.63, 4.033),
    'A_Coef': (0.7651, 8.456),
    'LKG': (4.519, 6.55)
}

# Label mapping
LABELS = {
    0: 'Kama',
    1: 'Rosa',
    2: 'Canadian'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        try:
            input_data = [float(request.form[f]) for f in FEATURES]
        except Exception:
            # Load dataset for preview (20 rows)
            import pandas as pd
            import os
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(BASE_DIR, 'data', 'seeds_dataset.csv')
            df = pd.read_csv(data_path)
            dataset_columns = df.columns.tolist()
            dataset_rows = df.head(20).values.tolist()
            return render_template('index.html', error='Input tidak valid!', features=FEATURES, feature_ranges=FEATURE_RANGES,
                                   dataset_columns=dataset_columns, dataset_rows=dataset_rows)
        # Preprocess
        X = scaler.transform([input_data])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        # Load dataset for preview (20 rows)
        import pandas as pd
        import os
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(BASE_DIR, 'data', 'seeds_dataset.csv')
        df = pd.read_csv(data_path)
        dataset_columns = df.columns.tolist()
        dataset_rows = df.head(20).values.tolist()
        return render_template(
            'result.html',
            features=FEATURES,
            values=input_data,
            label=LABELS[pred],
            proba=[f"{p*100:.1f}%" for p in proba],
            label_idx=pred,
            dataset_columns=dataset_columns,
            dataset_rows=dataset_rows
        )
    # Load dataset for preview (20 rows)
    import pandas as pd
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'data', 'seeds_dataset.csv')
    df = pd.read_csv(data_path)
    dataset_columns = df.columns.tolist()
    dataset_rows = df.head(20).values.tolist()
    dataset_rows_full = df.values.tolist()  # seluruh baris untuk tabel dan dropdown baris
    # Siapkan opsi dropdown untuk setiap fitur dari seluruh dataset (urutan sesuai dataset)
    feature_options = {f: [row[i] for row in dataset_rows_full] for i, f in enumerate(FEATURES)}
    return render_template('index.html', features=FEATURES, feature_ranges=FEATURE_RANGES,
                           dataset_columns=dataset_columns, dataset_rows=dataset_rows_full, feature_options=feature_options, dataset_rows_full=dataset_rows_full)

@app.route('/visualizations/<img>')
def show_img(img):
    # Serve visualizations
    from flask import send_from_directory
    return send_from_directory('../visualizations', img)

if __name__ == '__main__':
    app.run(debug=True)
