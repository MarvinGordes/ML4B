import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Laden Sie Ihre CSV-Dateien hier
data_file_4 = pd.read_csv("Akkordeon.csv")
data_file_2 = pd.read_csv("Katze.csv")
data_file_3 = pd.read_csv("Mixer.csv")
data_file_1 = pd.read_csv("Mikrowelle.csv")
data_file_1 = data_file_1.drop('time', axis=1)
data_file_2 = data_file_2.drop('time', axis=1)
data_file_3 = data_file_3.drop('time', axis=1)
data_file_4 = data_file_4.drop('time', axis=1)

# Definieren Sie die Beschriftungen
label_options = ["Mixer", "Katze", "Akkordeon", "Mikrowelle"]

# Erstellen Sie die Streamlit-App
st.title("Kannst du Katzen von Küchengeräten unterscheiden?")
st.write("Jeder Graph zeigt die Audiospur von entweder einer Katze, einem Mixer, einer Mikrowelle oder einem Akkordeon an. Weise jedem Graphen den Ursprung des Geräuschs zu und klicke unten auf den Button, um zu sehen, ob du alles richtig eingeordnet hast!")

# Konvertieren Sie die Daten in Graphen und zeigen Sie sie an
fig_1, ax_1 = plt.subplots()
ax_1.plot(data_file_1["seconds_elapsed"], data_file_1["dBFS"])
ax_1.set_title("Datensatz 1")
st.pyplot(fig_1)

fig_2, ax_2 = plt.subplots()
ax_2.plot(data_file_2["seconds_elapsed"], data_file_2["dBFS"])
ax_2.set_title("Datensatz 2")
st.pyplot(fig_2)

fig_3, ax_3 = plt.subplots()
ax_3.plot(data_file_3["seconds_elapsed"], data_file_3["dBFS"])
ax_3.set_title("Datensatz 3")
st.pyplot(fig_3)

fig_4, ax_4 = plt.subplots()
ax_4.plot(data_file_4["seconds_elapsed"], data_file_4["dBFS"])
ax_4.set_title("Datensatz 4")
st.pyplot(fig_4)

st.markdown("---")

label_1 = st.selectbox("Datensatz 1", options=label_options)
label_2 = st.selectbox("Datensatz 2", options=label_options)
label_3 = st.selectbox("Datensatz 3", options=label_options)
label_4 = st.selectbox("Datensatz 4", options=label_options)

st.markdown("---")

audio_file_1 = open("Mix.mp3", "rb").read()
audio_file_2 = open("Cat.mp3", "rb").read()
audio_file_3 = open("Microwave.mp3", "rb").read()
audio_file_4 = open("Accord.mp3", "rb").read()

st.audio(audio_file_1, format='audio/mp3', start_time=0)
st.button("Mixer")

st.audio(audio_file_2, format='audio/mp3', start_time=0)
st.button("Katze")

st.audio(audio_file_3, format='audio/mp3', start_time=0)
st.button("Mikrowelle")

st.audio(audio_file_4, format='audio/mp3', start_time=0)
st.button("Akkordeon")

st.markdown("---")

# Erstellen Sie einen Button zum Auswerten der Beschriftungen
if st.button("Auswerten"):
    # Definieren Sie die korrekten Antworten
    correct_labels = ["Mikrowelle", "Katze", "Mixer", "Akkordeon"]
    
    # Überprüfen Sie die Antworten des Benutzers
    user_labels = [label_1, label_2, label_3, label_4]
    num_correct = sum([1 for i in range(4) if user_labels[i] == correct_labels[i]])
    
    # Zeigen Sie das Ergebnis an
    st.write(f"Sie haben {num_correct} von 4 Datensätzen korrekt beschriftet.")