import json
import os
import numpy as np
import joblib
import onnxruntime as ort
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs

# Load model dan scaler saat cold start
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
session = ort.InferenceSession(os.path.join(MODEL_DIR, 'model_lstm.onnx'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))

INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Baca body request
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            
            # Parse JSON body
            data = json.loads(body)
            input_list = data.get('data', [])
            
            if not input_list or not isinstance(input_list, list):
                raise ValueError("Input harus berupa array angka")
            
            input_list = [float(x) for x in input_list]
            
            # Scaling & Reshape input untuk LSTM
            scaled_input = scaler.transform(np.array(input_list).reshape(-1, 1))
            
            # Pad atau truncate ke 60 timesteps
            timesteps = 60
            if len(scaled_input) < timesteps:
                # Pad dengan nilai pertama di awal
                pad_length = timesteps - len(scaled_input)
                padding = np.full((pad_length, 1), scaled_input[0][0])
                scaled_input = np.vstack([padding, scaled_input])
            elif len(scaled_input) > timesteps:
                scaled_input = scaled_input[-timesteps:]
            
            reshaped_input = scaled_input.reshape(1, timesteps, 1).astype(np.float32)
            
            # Prediksi dengan ONNX Runtime
            pred_scaled = session.run([OUTPUT_NAME], {INPUT_NAME: reshaped_input})[0]
            pred_original = scaler.inverse_transform(pred_scaled)
            result = round(float(pred_original[0][0]), 2)
            
            # Format response
            labels = [f"Hari {i+1}" for i in range(len(input_list))] + ["Prediksi Besok"]
            historical_data = input_list + [None]
            prediction_data = [None] * (len(input_list) - 1) + [input_list[-1], result]
            
            response = {
                'success': True,
                'prediction': result,
                'prediction_text': f'Rp {result:,.2f}',
                'labels': labels,
                'historical_data': historical_data,
                'prediction_data': prediction_data
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'success': False,
                'error': str(e)
            }).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
