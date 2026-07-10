from flask import Flask, request, jsonify, send_from_directory, session
import pickle
import numpy as np
import os
import sqlite3
from datetime import datetime

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend')
DB_PATH = os.path.join(os.path.dirname(__file__), 'predictions.db')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
app.secret_key = 'cereal-yield-prediction-secret-key-2026'

SITE_PASSWORD = 'agric2026'  # change this if you want a different password

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

model = pickle.load(open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'rb'))
le_area = pickle.load(open(os.path.join(MODELS_DIR, 'label_encoder_area.pkl'), 'rb'))
le_item = pickle.load(open(os.path.join(MODELS_DIR, 'label_encoder_item.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb'))


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop TEXT,
            country TEXT,
            year INTEGER,
            rainfall REAL,
            temperature REAL,
            pesticide REAL,
            area_harvested REAL,
            predicted_yield REAL,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()


def require_login():
    return session.get('logged_in') is True


@app.route('/')
def root():
    if require_login():
        return send_from_directory(FRONTEND_DIR, 'dashboard.html')
    return send_from_directory(FRONTEND_DIR, 'login.html')


@app.route('/predict-page')
def predict_page():
    if not require_login():
        return send_from_directory(FRONTEND_DIR, 'login.html')
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/dashboard.html')
def dashboard_page():
    if not require_login():
        return send_from_directory(FRONTEND_DIR, 'login.html')
    return send_from_directory(FRONTEND_DIR, 'dashboard.html')


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    if data.get('password') == SITE_PASSWORD:
        session['logged_in'] = True
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Incorrect password'}), 401


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


@app.route('/api/session', methods=['GET'])
def check_session():
    return jsonify({'logged_in': require_login()})


@app.route('/crops', methods=['GET'])
def get_crops():
    return jsonify(list(le_item.classes_))


@app.route('/countries', methods=['GET'])
def get_countries():
    return jsonify(sorted(list(le_area.classes_)))


@app.route('/api/stats', methods=['GET'])
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute('SELECT COUNT(*) FROM predictions')
    total_predictions = cur.fetchone()[0]
    conn.close()

    return jsonify({
        'model_r2': 0.9602,
        'model_name': 'XGBoost Regressor',
        'training_records': 13778,
        'countries_covered': 101,
        'crops_covered': 4,
        'total_predictions_made': total_predictions
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT 20')
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return jsonify(rows)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        country = data['country']
        crop = data['crop']
        year = int(data['year'])
        rainfall = float(data['rainfall'])
        pesticide = float(data['pesticide'])
        temp = float(data['temperature'])
        area = float(data['area_harvested'])

        area_encoded = le_area.transform([country])[0]
        item_encoded = le_item.transform([crop])[0]

        scaled_values = scaler.transform([[rainfall, pesticide, temp, area]])[0]
        rainfall_s, pesticide_s, temp_s, area_s = scaled_values

        features = np.array([[area_encoded, item_encoded, year, rainfall_s, pesticide_s, temp_s, area_s]])

        prediction = model.predict(features)[0]
        predicted_yield = round(float(prediction), 2)

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            '''INSERT INTO predictions
               (crop, country, year, rainfall, temperature, pesticide, area_harvested, predicted_yield, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (crop, country, year, rainfall, temp, pesticide, area, predicted_yield, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

        return jsonify({
            'predicted_yield_hg_ha': predicted_yield,
            'crop': crop,
            'country': country,
            'year': year
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)