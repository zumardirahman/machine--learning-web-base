from flask import render_template, request, jsonify
from pprint import pprint
import numpy as np
import model as model

def index():
    return render_template('index.html') 

def predict():
    try:
        data = request.json  # Mendapatkan input dari frontend (JSON)
        input_data = np.array([data['Merk'], data['RAM_GB'], data['Storage_GB'], data['Kamera_MP'], data['Baterai_mAh']]).reshape(1, -1)

        pprint(input_data)

        # Melakukan prediksi
        prediksi = model.model_proses.predict(input_data)
        return jsonify({'prediksi_harga': round(prediksi[0], 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)})