import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

# Definieren der Abtastrate
sr = 22050


# Laden der CSV-Dateien als Numpy-Arrays
water_data1 = np.loadtxt('knock3.csv', delimiter=',', skiprows=1, usecols=2)
water_data2 = np.loadtxt('knock4.csv', delimiter=',', skiprows=1, usecols=2)
knock_data1 = np.loadtxt('knock5.csv', delimiter=',', skiprows=1, usecols=2)
knock_data2 = np.loadtxt('knock6.csv', delimiter=',', skiprows=1, usecols=2)

water_data1=water_data1.astype(int)
water_data2=water_data2.astype(int)
knock_data1=knock_data1.astype(int)
knock_data2=knock_data2.astype(int)

water_data1=np.ndarray(water_data1)
water_data2=np.ndarray(water_data2)
knock_data1=np.ndarray(knock_data1)
knock_data2=np.ndarray(knock_data2)


# Extrahieren der Merkmale aus den Audiodateien
def extract_features(data, sr):
    features = []
    for signal in data:
        # Berechnen der Mel-Frequenz-Cepstrum-Koeffizienten
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        # Skalieren der Merkmale auf den Bereich [-1, 1]
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        features.append(mfcc_scaled)
    return np.array(features)



# Extrahieren der Merkmale für jedes Signal und jede Klasse
water_features1 = extract_features(water_data1, sr)
water_features2 = extract_features(water_data2, sr)
knock_features1 = extract_features(knock_data1, sr)
knock_features2 = extract_features(knock_data2, sr)

# Zusammenführen aller Merkmale in einem Numpy-Array
X = np.vstack((water_features1, water_features2, knock_features1, knock_features2))
y = np.concatenate((np.zeros(len(water_features1) + len(water_features2)), np.ones(len(knock_features1) + len(knock_features2))))

# Erstellen des Trainings- und Testdatensatzes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Erweitern der Dimensionen des Inputs
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

# Beispiel für das Erstellen und Trainieren eines CNNs mit Keras
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test,y_test))