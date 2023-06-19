import csv
import random
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def process_csv_file(csv_file, sample_size, num_samples):
    samples = []
    used_samples = set()

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Überspringe die Header-Zeile

        lines = list(reader)
        lines_without_first_column = [line[1:] for line in lines]

        total_lines = len(lines_without_first_column)
        max_start_index = total_lines - sample_size

        if max_start_index <= 0:
            print("Nicht genügend Zeilen in der CSV-Datei.")
            return

        while len(samples) < num_samples:
            start_index = random.randint(0, max_start_index)
            sample = lines_without_first_column[start_index: start_index + sample_size]
            sample_str = str(sample)

            if sample_str not in used_samples:
                label = None
                if 'Knocking' in csv_file:
                    label = 'Knock'
                elif 'Water' in csv_file:
                    label = 'Water'
                samples.append((sample, label))
                used_samples.add(sample_str)

    print(f"Anzahl der einzigartigen Samples: {len(samples)}")
    return samples


# Beispielaufruf der Funktion
sample_size = 50
num_samples = 300

knock_data = process_csv_file('Knocking.csv', sample_size, num_samples)
water_data = process_csv_file('Water.csv', sample_size, num_samples)

# Mische die Daten in einer zufälligen Reihenfolge
dataset = knock_data + water_data
random.shuffle(dataset)

# Streamlit-App
st.title("Graphen der Datensätze")
st.write("Dies ist eine Streamlit-App, die Graphen der Datensätze anzeigt.")

plots_per_row = 3
plot_size = (6, 4)

# Iteriere über die Samples und erstelle Plots
for i, sample in enumerate(dataset):
    # Erstelle einen neuen Plot für jedes Sample
    if i % plots_per_row == 0:
        # Neue Zeile für die Plots
        st.write("---")
        fig, axs = plt.subplots(1, plots_per_row, figsize=(plot_size[0] * plots_per_row, plot_size[1]))
        plt.tight_layout(pad=4.0)

    # Extrahiere die Daten aus dem Sample
    data = np.array(sample[0], dtype=float)
    seconds_elapsed = data[:, 0]
    dBFS = data[:, 1]

    # Plot erstellen
    ax = axs[i % plots_per_row]
    ax.plot(seconds_elapsed, dBFS)
    ax.set_xlabel('Zeit (s)')
    ax.set_ylabel('dBFS')
    ax.set_title(f'Sample {i+1}')

    # Ausgabe des Labels
    label = sample[1]
    st.write(f"Label: {label}")

    # Überprüfe, ob dies der letzte Plot in der Zeile ist
    if (i + 1) % plots_per_row == 0 or (i + 1) == len(dataset):
        st.pyplot(fig)
        plt.close(fig)

# Ausgabe der ersten 10 Einträge von dataset
st.write("Inhalte der ersten 10 Einträge von dataset:")
for i, sample in enumerate(dataset[:10]):
    st.write(f"Sample {i+1}:")
    st.write(f"Daten: {sample[0]}")
    st.write(f"Label: {sample[1]}")
    st.write("---")
