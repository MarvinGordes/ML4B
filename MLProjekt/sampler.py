import csv
import random
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def process_csv_file(csv_file, sample_size, num_samples, plots_per_row=3, plot_size=(6, 4)):
    # Array zur Speicherung der Samples
    samples = []

    # Set zur Verfolgung bereits verwendeter Samples
    used_samples = set()

    # CSV-Datei einlesen und erste Spalte entfernen
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Überspringe die Header-Zeile

        # Liste aller Zeilen
        lines = list(reader)

        # Extrahiere die Daten ohne die erste Spalte
        lines_without_first_column = [line[1:] for line in lines]

        total_lines = len(lines_without_first_column)

        # Anzahl der möglichen Startpunkte für ein Sample
        max_start_index = total_lines - sample_size

        # Überprüfe, ob genügend Zeilen vorhanden sind
        if max_start_index <= 0:
            print("Nicht genügend Zeilen in der CSV-Datei.")
            return

        # Erstelle die Samples
        while len(samples) < num_samples:
            # Zufälliger Startpunkt für ein Sample
            start_index = random.randint(0, max_start_index)

            # Extrahiere die Zeilen für das Sample
            sample = lines_without_first_column[start_index: start_index + sample_size]

            # Konvertiere das Sample in eine Zeichenkette, um es im Set zu speichern
            sample_str = str(sample)

            # Überprüfe, ob das Sample bereits verwendet wurde
            if sample_str not in used_samples:
                # Füge das Sample zum Array hinzu
                samples.append(sample)

                # Füge das Sample zur Menge der verwendeten Samples hinzu
                used_samples.add(sample_str)

    # Ausgabe der Anzahl der erstellten Samples
    print(f"Anzahl der einzigartigen Samples: {len(samples)}")

    # Iteriere über die Samples und erstelle Plots
    for i, sample in enumerate(samples):
        # Erstelle einen neuen Plot für jedes Sample
        if i % plots_per_row == 0:
            # Neue Zeile für die Plots
            st.write("---")
            fig, axs = plt.subplots(1, plots_per_row, figsize=(plot_size[0] * plots_per_row, plot_size[1]))
            plt.tight_layout(pad=4.0)

        # Extrahiere die Daten aus dem Sample
        data = np.array(sample, dtype=float)
        seconds_elapsed = data[:, 0]
        dBFS = data[:, 1]

        # Plot erstellen
        ax = axs[i % plots_per_row]
        ax.plot(seconds_elapsed, dBFS)
        ax.set_xlabel('Zeit (s)')
        ax.set_ylabel('dBFS')
        ax.set_title(f'Sample {i+1}')

        # Überprüfe, ob dies der letzte Plot in der Zeile ist
        if (i + 1) % plots_per_row == 0 or (i + 1) == len(samples):
            st.pyplot(fig)
            plt.close(fig)

    # Ausgabe der ersten 10 Einträge von samples
    st.write("Inhalte der ersten 10 Einträge von samples:")
    for i, sample in enumerate(samples[:10]):
        st.write(f"Sample {i+1}:")
        st.write(sample)
        st.write("---")


# Beispielaufruf der Funktion
csv_file = 'Water.csv'
sample_size = 50
num_samples = 300

process_csv_file(csv_file, sample_size, num_samples)