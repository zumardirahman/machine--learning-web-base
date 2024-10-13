from flask import Flask, render_template, request, jsonify
import controller as controller


# WEB
app = Flask(__name__)


# FRONTEND
# Halaman Web utama
@app.route('/')
def index_route():
    return controller.index()


# BACKEND
# API untuk prediksi harga handphone
@app.route('/predict', methods=['POST'])
def predict_route():
    return controller.predict()



if __name__ == '__main__':
    app.run(debug=True)
