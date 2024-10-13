# Import library yang diperlukan
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Data training handphone (disesuaikan dengan kasus)
data_hp = {
    'Merk': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'RAM_GB': [4, 6, 8, 4, 8, 6, 12, 8, 4, 6],
    'Storage_GB': [64, 128, 256, 64, 128, 256, 64, 128, 256, 128],
    'Kamera_MP': [12, 16, 24, 12, 48, 12, 48, 16, 12, 24],
    'Baterai_mAh': [3000, 3500, 4000, 3000, 4500, 3500, 5000, 4000, 3000, 4500],
    'Harga': [5000, 10000, 15000, 6000, 16000, 11000, 20000, 15000, 7000, 13000]
}

df_hp = pd.DataFrame(data_hp)

# Memisahkan fitur dan target
X = df_hp[['Merk', 'RAM_GB', 'Storage_GB', 'Kamera_MP', 'Baterai_mAh']]
y = df_hp['Harga']

# Melatih model regresi linier
model_hp = LinearRegression()
model_hp.fit(X, y)

# Menyimpan model menggunakan pickle
with open('model_hp.pkl', 'wb') as f:
    pickle.dump(model_hp, f)
