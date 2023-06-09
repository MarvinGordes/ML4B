import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report
import numpy as np

# WICHTIG: Damit das hier auf deinem LocalHost läuft, in Zeile 75 das "Final/" aus dem Dateipfad nehmen, das brauchts nur wenn der Code über Github läuft
# Kurzer Rundown der wichtigsten Methoden falls du hier etwas ausarbeiten willst:
#
# Größere Schriftzüge: st.title("Lorem Ipsum"), st.header("Lorem Ipsum"), st.subheader("Lorem Ipsum")
#
# Text: st.write("Für Einzeiler"), st.markdown("Für alles mit Zeilenumbruch")  -> Absätze hab ich bisher einfach mit mehreren markdowns gemacht, evtl gehts auch hübscher, kA
#
# Ausklappbare Textfelder: with st.expander("Titel"):
#                            st.write("Einrücken nicht vergessen")
#                            
# Medien: image1 = "beispiel.png"  -> st.image(image1), video1 = "beispiel.mp4" -> st.video(video1) 
#        Falls es hier Probleme gibt, liegts wahrscheinlich am Dateipfad, aber da du das sowieso auf dem LocalHost laufen lassen wirst, ist das nicht besonders kompliziert
#
# Wenn du Sachen nebeneinander packen willst:  col1,col2 = st.columns(2)
#                                            with col1:
#                                                st.write("Wieder mit Einrücken, für col2 dasselbe")
#
# Wenn du noch schauen willst, was es sonst so gibt: https://docs.streamlit.io/library/cheatsheet


def home():
    st.header("Willkommen auf unserem Audio-Erkennungs-Tool!")
    st.markdown("Wir verwenden ein Machine-Learning-Modell, um verschiedene Arten von Tonaufnahmen unterscheiden zu können. Ziel ist es, das Modell und die Website noch weiterzuentwickeln, um mittelfristig gehörlosen Menschen zu helfen, auf Geräusche aufmerksam zu werden. ")
    # Füge hier den Inhalt für die Home-Seite hinzu

def page1():
    st.title("Hintergrund")
    with st.expander("Die Entwickler"):
        col1,col2 = st.columns(2)
        with col1:
            #st.image(image1)
            st.write("bild 1")
        with col2:
           # st.image(image1)
           st.write("bild 2")
        st.write("Lorem Ipsum")
    with st.expander("Unsere Idee"):
        st.markdown("Unser Ziel ist es, gehörlosen Menschen mit unserer Lösung eine kostengünstige und flexible Möglichkeit zu bieten, auf relevante Geräusche aufmerksam gemacht zu werden.  Es existieren bestehende Lösungen auf den verbreiteten Smartphone-Betriebssystemen (vgl. https://www.netzwelt.de/anleitung/188298-android-so-erkennt-handy-geraeusche-alarmiert-euch.html und https://www.giga.de/tipp/geraeuscherkennung-am-iphone-so-gehts/). Wenn das Handy aber in der Hosentasche oder im Rucksack ist, lässt sich damit die Wohnung nicht zuverlässig abdecken. Außerdem sind die erkennbaren Geräusche limitiert.  Die jetzige Version zeigt, dass es mit vergleichsweise wenigen selbst aufgenommenen Trainingsdaten möglich ist, ein gut funktionierendes Machine Learning Modell zu trainieren. Zukünftig sollen neben laufendem Wasser, Klopfen und Mikrowellen weitere Geräusche in die Liste der erkennbaren Geräusche mit aufgenommen werden.  Zudem soll die Anwendung live laufen können und kostengünstige, dauerhaft installierbarre Hardware wie den Raspberry Pi unterstützen. Darüberhinaus soll eine Anleitung erstellt werden, wie auch Menschen ohne Programmiererfahrung das Modell für ihre konkreten Geräusche (z.B. ihre Türklingel) anpassen können. ")
    with st.expander("Entwicklungs-Log"):
        st.write("Lorem Ipsum")

def page2():
    st.header("Wie können Sie unser Modell ausprobieren?")
    
    st.subheader("1. Nehmen Sie ihr Sample auf")
    st.markdown("Laden Sie sich die Sensor-Logger App für Ihr Smartphone herunter: ")
    st.markdown("iOS: https://apps.apple.com/us/app/sensor-logger/id1531582925")
    st.markdown("Android: https://play.google.com/store/apps/details?id=com.kelvin.sensorapp&pli=1")          
    st.markdown("Dann können Sie eine Tonaufnahme von Türklopfen, laufendem Wasser oder einer Mikrowelle aufnehmen.")
    
    st.subheader("2. Laden Sie ihr Sample auf der nächsten Seite hoch:")
    #st.video(placeholder)
    st.write("Hier Video als Demonstration? Also Bildschirmaufnahme mit extrahieren, raussuchen der .csv und hochladen auf unserer App")
    
    st.subheader("3. Visualisieren Sie ihr Sample")
    st.markdown("Über die nächsten zwei Buttons können Sie ihr Sample als Rohdaten oder Graph ausgeben lassen.")
    
    st.subheader("4. Werten Sie ihr Sample aus")
    st.markdown("Über den letzten Button wird Ihr Sample von unserem RandomForest Modell in eine der drei Kategorien eingeteilt, mit denen es trainiert hat")
  

def page3():
    
    st.title("Teste unser Modell!")
    
    # Modell laden

    model = open('Final/forest10-12000.pkl', 'rb')
    classifier = pickle.load(model)

    # Upload-Funktion 

    userSample = st.file_uploader("Lade deine .csv Datei hoch!")
   

    if userSample is not None:
        file = pd.read_csv(userSample)
        file.drop(['time'], axis=1, inplace=True) 
        
        with st.expander("Rohdaten anzeigen"):
            st.write(file)
             
        with st.expander("Graph anzeigen"):
            df= file
            fig, ax = plt.subplots()
            ax.plot(df['seconds_elapsed'], df['dBFS'])
            ax.set_xlabel('Zeit in Sek')
            ax.set_ylabel('Lautstärke in dB')
            ax.set_title('Dein Sample')
            st.pyplot(fig)
        # Vorbereiten der hochgeladenen Daten
        
        
        file['dBFS'] = file['dBFS'].round(decimals = 0)
        file['Label'] = 'Blank'
        
    # Einordnungs-Funktion

    def prediction(input):
        data = []  
        data.append(pd.DataFrame({"dBFS_Varianz": input["dBFS"].var(),"dBFS_STD" : input["dBFS"].std(), "dBFS_mean" : input["dBFS"].mean(),
            "dBFS_min" : input["dBFS"].min(), "dBFS_max" : input["dBFS"].max(), "dBFS_absMax" : input["dBFS"].abs().max(), "dBFS_sum" : input["dBFS"].sum(), "dBFS_median" : input["dBFS"].median(), "Label" : input["Label"]}))
        
        data[0] = data[0].drop('Label', axis=1)
        prediction = classifier.predict(data[0])
        prediction = prediction[0]
        st.write(data[0].head(1)) 
        return prediction

    # Einordnungs-Button für User

    if st.button("Einordnung durch unser Modell"):
        st.subheader("Unser Modell hält dein Sample für: " + prediction(file))
                

def page4():
    st.title("Nächste Schritte")
    with st.expander("Modell"):
        st.write("Lorem Ipsum")
    with st.expander("Testdaten"):
        st.write("Lorem Ipsum")
    with st.expander("App"):
        st.write("Lorem Ipsum")
    # Füge hier den Inhalt für die Seite 4 hinzu
    
def page5():
    st.title("Feedback Section")
    name = st.text_input("Name")
    email = st.text_input("E-Mail")
    message = st.text_area("Nachricht")
    submit = st.button("Absenden")

    # Verarbeitung des Formulars
    if submit:
        if name and email and message:
            # Hier können Sie den Code einfügen, um das Formular zu verarbeiten (z. B. E-Mail senden, Datenbank speichern, usw.)
            st.success("Vielen Dank für Ihre Nachricht!")
        else:
            st.warning("Bitte füllen Sie alle Felder aus.")
    


# Definiere das Menü und die zugehörigen Seiten
pages = {
    "Startseite": home,
    "Hintergrund": page1,
    "Anleitung": page2,
    "Teste unser Modell": page3,
    "Nächste Schritte": page4,
    "Feedback": page5
}

# Streamlit-App
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Gehe zu", list(pages.keys()))
    page = pages[selection]
    page()

if __name__ == "__main__":
    main()
