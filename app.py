import numpy as np
import joblib
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder='Template')

model = load_model('Model/model_lstm.keras')
scaler = joblib.load('Model/scaler.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_input = request.form['data']
        input_list = [float(x.strip()) for x in raw_input.split(',')]
        
        # Scaling & Reshape input untuk LSTM
        scaled_input = scaler.transform(np.array(input_list).reshape(-1, 1))
        reshaped_input = scaled_input.reshape(1, len(scaled_input), 1)

        # Prediksi & balikan angka ke skala asli
        pred_scaled = model.predict(reshaped_input)
        pred_original = scaler.inverse_transform(pred_scaled)
        result = round(float(pred_original[0][0]), 2)

        # Format label & dataset untuk Grafik Chart.js
        labels = [f"Hari {i+1}" for i in range(len(input_list))] + ["Prediksi Besok"]
        
        # Line 1: Data historis
        historical_data = input_list + [None]
        
        # Line 2: Garis sambungan ke hasil prediksi (garis putus-putus)
        prediction_data = [None] * (len(input_list) - 1) + [input_list[-1], result]

        return render_template(
            'index.html',
            prediction_text=f'Rp {result:,.2f}',
            labels=labels,
            historical_data=historical_data,
            prediction_data=prediction_data,
            raw_input=raw_input
        )
    except Exception as e:
        return render_template('index.html', error_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)