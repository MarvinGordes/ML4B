import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import csv
import random
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def process_csv_file(csv_file, sample_size, num_samples, label):
    samples = []
    used_samples = set()

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Überspringe die Header-Zeile

        lines = list(reader)
        lines_without_first_column = [line[2:] for line in lines]

        total_lines = len(lines_without_first_column)
        max_start_index = 5000 - sample_size

        if max_start_index <= 0:
            print("Nicht genügend Zeilen in der CSV-Datei.")
            return

        while len(samples) < num_samples:
            start_index = random.randint(0, max_start_index)
            sample = lines_without_first_column[start_index: start_index + sample_size]
            sample = [[round(float(value)) for value in row] for row in sample]  # Runde alle Werte auf ganze Zahlen
            sample_str = str(sample)

            if label == 'Knocking':
                label = 'Knock'
            elif label == 'Water':
                label = 'Water'
            samples.append((sample, label))
            used_samples.add(sample_str)

    print(f"Anzahl der einzigartigen Samples: {len(samples)}")
    return samples

# Beispielaufruf der Funktion
sample_size = 50
num_samples = 1000

knock_data = process_csv_file('Knocking.csv', sample_size, num_samples, 'Knocking')
water_data = process_csv_file('Water.csv', sample_size, num_samples, 'Water')

# Mische die Daten in einer zufälligen Reihenfolge
dataset = knock_data + water_data
random.shuffle(dataset)

# Gib die ersten 10 Einträge des datasets aus
print('Erste 10 Einträge des datasets:')
for entry in dataset[:10]:
    print(entry)

# Konvertiere die Features in numerische Werte und richte die Dimensionen richtig aus
features = np.array([sample[0] for sample in dataset], dtype=float)
features = features.reshape(features.shape[0], -1)  # Hier wird die Dimension auf 2 angepasst
labels = np.array([sample[1] for sample in dataset])

# Aufteilen der Daten in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Erstelle den KNN-Klassifikator
knn = KNeighborsClassifier(n_neighbors=3)

# Trainiere den Klassifikator
knn.fit(X_train, y_train)

# Vorhersagen für die Testdaten machen
y_pred = knn.predict(X_test)

# Bewertung der Genauigkeit des Klassifikators
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print('Genauigkeit:', accuracy)
print('Präzision:', precision)
print('Recall:', recall)


# Berechne die Konfusionsmatrix
cm = confusion_matrix(y_test, y_pred)

# Gib die Konfusionsmatrix aus
print('Konfusionsmatrix:')
print(cm)
