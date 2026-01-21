# 1. IMPOR PUSTAKA
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# 2. MENYIAPKAN DATA
# [cite_start]Membaca file CSV (sesuai format asli di materi) [cite: 10, 303]
data = pd.read_csv('databeasiswa.csv', header=None)
dataset = np.asarray(data)

# Memisahkan input (X) dan output (Y)
# X = Data kolom 1 dan 2 (IPK, Tingkat Kemiskinan) mulai dari baris ke-1 (skip header)
X = dataset[1:, 1:3].astype(float)
# Y = Data kolom 3 (Status Beasiswa)
Y = dataset[1:, 3:4].astype(float)

# 3. MEMBUAT MODEL JST (EKSPERIMEN MODIFIKASI)
def create_model():
    model = Sequential()
    
    # [cite_start]Layer Input & Hidden 1: Diperbesar ke 20 neuron (sebelumnya 12) [cite: 11]
    model.add(Dense(20, input_dim=2, activation='relu'))
    
    # Hidden Layer 2: Diperbesar ke 10 neuron (sebelumnya 8)
    model.add(Dense(10, activation='relu'))
    
    # Output Layer: 1 neuron sigmoid (0/1)
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 4. KONFIGURASI VALIDASI SILANG (CROSS VALIDATION)
# Seed untuk reproduktifitas
seed = 7
np.random.seed(seed)

# Wrapper Keras untuk Scikit-Learn
# [cite_start]EKSPERIMEN: Epoch dinaikkan jadi 300 (sebelumnya 150) agar belajar lebih lama [cite: 303]
model = KerasClassifier(build_fn=create_model, epochs=300, batch_size=5, verbose=0)

# Menggunakan 10-Fold Cross Validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# 5. EKSEKUSI DAN HASIL
print("Memulai eksperimen pelatihan model...")
results = cross_val_score(model, X, Y, cv=kfold)

print("\n=== HASIL EKSPERIMEN ===")
print(f"Akurasi per Fold: {results}")
print(f"Rata-rata Akurasi: {results.mean()*100:.2f}%")
print(f"Standar Deviasi: {results.std():.4f}")