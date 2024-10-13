from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load model yang sudah dilatih sebelumnya
with open('model_hp.pkl', 'rb') as f:
    model_hp = pickle.load(f)

# Halaman utama
@app.route('/')
def index():
    return render_template('index.html')  # Mengarahkan ke halaman HTML

# API untuk prediksi harga handphone
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Mendapatkan input dari frontend (JSON)
        input_data = np.array([data['Merk'], data['RAM_GB'], data['Storage_GB'], data['Kamera_MP'], data['Baterai_mAh']]).reshape(1, -1)

        # Melakukan prediksi
        prediksi = model_hp.predict(input_data)
        return jsonify({'prediksi_harga': round(prediksi[0], 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
