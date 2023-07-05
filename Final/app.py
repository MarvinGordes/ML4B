import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report
import numpy as np

# WICHTIG: Damit das hier auf deinem LocalHost läuft, in Zeile 75 das "Final/" aus dem Dateipfad nehmen, das brauchts nur wenn der Code über Github läuft
# Kurzer Rundown der wichtigsten Methoden:
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
    st.header("Willkommen auf unserem Audio-Klassifizierungs-Tool!")
    st.markdown("Wir verwenden ein Machine-Learning-Modell, um verschiedene Arte von Geräuschen voneinander unterscheiden zu können. Unser Ziel ist es, das Machine-Learning-Modell und diese Anwendung so weiterzuentwickeln, dass sie gehörlosen Menschen dabei helfen kann, auf relevante Geräusche aufmerksam zu werden. ")
    # Füge hier den Inhalt für die Home-Seite hinzu


def page1():
    st.header("So können Sie unser Audio-Klassifizierungs-Modell nutzen")
    
    st.subheader("1. Laden Sie die Sensor-Logger App für Ihr Smartphone herunter")
    st.markdown("iOS: https://apps.apple.com/us/app/sensor-logger/id1531582925")
    st.markdown("Android: https://play.google.com/store/apps/details?id=com.kelvin.sensorapp&pli=1")          
    st.subheader("2. Nehmen Sie ein von unserem Modell unterstütztes Geräusch auf")
    st.markdown("Aktivieren Sie auf dem Reiter 'Logger' 'Microphone' und erstellen Sie eine Tonaufnahme von **Türklopfen, laufendem Wasser oder einer Mikrowelle** mittels 'Start Recording'.")
    st.markdown("***Bitte beachten Sie, dass aktuell keine weiteren Geräusche unterstützt werden. Andere Geräusche werden daher fälschlicherweise auch als Türklopfen, laufendes Wasser oder als Mikrowelle erkannt.***")

    
    st.subheader("3. Laden Sie die Datei hoch:")
    st.markdown("Klicken Sie in der Sensor Logger App auf den Reiter 'Recordings', klicken Sie auf die von Ihnen gemachte Aufnahme, wählen Sie 'Export' und dann 'CSV in Zip File' -> Export Recording. Speichern Sie die Daten an einem passenden Ort und entzippen Sie die Datei. Uploaden Sie nun aus dem entstandenen Ordner die Datei 'Microphone.csv' unter diesem Absatz (siehe 'Drag and drop file here'). Klicken Sie auf 'Vorhersage durch unser Modell', um eine Einordnung Ihrer Aufnahme zu erhalten.")
    #st.video(placeholder)
    #st.write("Hier Video als Demonstration? Also Bildschirmaufnahme mit extrahieren, raussuchen der .csv und hochladen auf unserer App")
    

    # Modell laden

    model = open('forest10-13000.pkl', 'rb')
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

    if st.button("Vorhersage durch unser Modell"):
        st.subheader("Unsere Vorhersage: " + prediction(file))
                

    #st.subheader("4. Visualisieren Sie ihre Daten")
    #st.markdown("Wenn Sie möchten, können Sie sich mit den entsprechenden Schaltern die Rohdaten Ihrer CSV-Datei anzeigen und einen Graph für Ihre Daten ausgeben lassen.")
    
    #st.subheader("5. Werten Sie Ihre Daten aus")
    #st.markdown("Über den letzten Button wird Ihr Sample von unserem RandomForest Modell in eine der drei Kategorien 'Wasser', 'Klopfen' oder 'Mikrowelle' eingeteilt")

def page2():
    st.title("Hintergrund")
    
    with st.expander("Unsere Idee"):
        st.markdown("**Unser Ziel ist es, gehörlosen Menschen mit unserer Lösung eine kostengünstige und flexible Möglichkeit zu bieten, auf relevante Geräusche aufmerksam gemacht zu werden.**  Es existieren bestehende Lösungen auf den verbreiteten Smartphone-Betriebssystemen (vgl. https://www.netzwelt.de/anleitung/188298-android-so-erkennt-handy-geraeusche-alarmiert-euch.html und https://www.giga.de/tipp/geraeuscherkennung-am-iphone-so-gehts/). Leider lässt sich die gesamte Wohnung mit nur einem Smartphone nicht zuverlässig abdecken. Wenn also z.B. im Nebenraum aus Versehen Wasser läuft, reicht diese Lösung nicht. Außerdem funktioniert die Erkennung schlechter, wenn das Smartphone in der Hosentasche oder in der Tasche  ist.  Die erkennbaren Geräusche sind zudem limitiert.  **Die jetzige Version unserer Anwendung ist ein Proof of Concept**, der zeigt, dass es mit vergleichsweise wenigen selbst aufgenommenen Trainingsdaten möglich ist, ein gut funktionierendes Machine Learning Modell zu trainieren.")
    with st.expander("Unser Modell"):
        st.markdown("Bei unserem Machine-Learning-Modell handelt es sich um einen **Random Forest Classifier**. Random Forest bedeutet hierbei, dass der Algorithmus die Ergebnisse **mehrere Entscheidungsbäume** verwendet, um eine Entscheidung zu treffen. Classifier bedeutet hier, dass es sich um eine **Klassifikationsaufgabe** handelt (also entschieden wird, ob es sich z.B. um die Kategorie 'Wasser' oder 'Klopfen' handelt). Wir verwenden diesen Algorithmus, weil er vergleichsweise einfach zu implementieren ist und benötigt nur wenig Trainingszeit benötigt. Darüber hinaus ist hier ersichtlich, welche Features welche Bedeutung für das Modell hat. ")
        st.markdown("Den Datensatz für das Modell haben die Autoren mittels Sensor Logger App selbst generiert. Da als Sensor nur Smartphone-Mikrophone verwendet wurden, waren zunächst Dezibel-Zeitreihen das einzige Feature. Im Rahmen des **Feature-Engineering** wurden daher zur Verbesserung des Modells **weitere Werte wie die Varianz, Mittelwerte oder Maximum-Werte** gebildet. ")
        st.markdown("Um trotz der relativ geringen Datenmenge möglichst gute Ergebnisse erzielen zu können, haben wir eine Funktion geschrieben, um sogenanntes **Windowing** zu implementieren. Somit werden **unterschiedliche, aber überschneidende Zeitreihen aus den Daten herausgeschnitten**. Diese werden dann anschließend in den Test- und Trainingsdaten verwendet.")
        code = '''def split_dataframe(dataframe, sampleSize, sampleCount, max_attempts=100):
    df_length = len(dataframe)
    dataframes = []
    unique_ids = set()
    attempts = 0

    # While Schleife definiert durch gewünschte Sample-Menge

    while len(dataframes) < sampleCount and attempts < max_attempts:
        
        # Datensatzgröße prüfen
        
        max_start_idx = df_length - sampleSize
        if max_start_idx < 0:
            break
        
        # Potentielle Eckpunkte für Sample finden 

        start_idx = random.randint(0, max_start_idx)
        end_idx = start_idx + sampleSize

        if end_idx > df_length:
            start_idx = df_length - sampleSize

        # Sample generieren und Hash zuweisen
        
        df_slice = dataframe.iloc[start_idx:end_idx].copy()
        df_id = hash(df_slice.to_string())

        # Doppelte Samples filtern
        
        if df_id in unique_ids:
            attempts += 1
            continue

        # Sample in Liste aufnehmen
        
        dataframes.append(df_slice)
        unique_ids.add(df_id)
        attempts = 0

    return dataframes

# Auswahl der Trainingsdaten-Länge / -Menge: 
# 10 Zeilen ~ 1 Sekunde
# max Sample Menge = Zeilen in csv Datei'''
        st.code(code, language='python')

    with st.expander("Über die Entwickler"):
        col1,col2 = st.columns(2)
        with col1:
            #st.image(image1)
            st.write("bild 1")
            st.markdown("Martin Sigl studiert Wirtschaftsinformatik im Bachelor an der Friedrich-Alexander-Universität Erlangen-Nürnberg im 5. Fachsemester. Er beschäftigt sich seit Kurzem auch mit Machine-Learning. Martin möchte dazu beitragen, dass diese und andere spannende Technologien nicht nur dort eingesetzt werden, wo sie wirtschaftlich möglichst rentabel sind, sondern auch da, wo sie anderweitig Nutzen schaffen.") 
        with col2:
           # st.image(image1)
           st.write("bild 2")
           st.write("Lorem Ipsum")


def page3():
    st.title("Nächste Schritte")
    st.markdown("Wir haben gerade erst angefangen! Hier sind unsere Pläne dafür, wie wir unser Modell und unsere Anwendung weiterentwickeln wollen:")
    with st.expander("Modell"):
        st.write("Zukünftig sollen neben laufendem Wasser, Klopfen und Mikrowellen **weitere Geräusche in die Liste der erkennbaren Geräusche mit aufgenommen** werden. Welche Geräusche priorisiert bearbeitet werden, soll auf Basis von Feedback aus der Gehörlosen-Community erarbeitet werden.")
    with st.expander("Hardware"):
        st.write("Die Anwendung soll schon bald auch **live auf kostengünstiger Hardware wie den Raspberry Pi** laufen können. Diese wird es ermöglichen, dass garantiert der gewünschte räumliche Bereich von unserer Geräuscherkennung auch vom Mikrophon abgedeckt werden kann. Somit muss man nicht mehr darauf achten, sein Handy nicht im Nebenraum oder in der Tasche liegen zu lassen.")
    with st.expander("Dokumentation"):
        st.write("Um es zu ermöglichen, dass auch ohne unsere Hilfe weitere Geräusche bedarfsgerecht unserem Modell hinzugefügt werden können, werden wir eine Anleitung schreiben. Mit Hilfe der Anleitung sollen **auch Menschen ohne Programmiererfahrung das Modell für ihre Geräusche (z.B. ihre Türklingel) trainieren können.** ")
    
def page4():
    st.title("Feedback")
    st.markdown("Wenn Sie Anregungen oder Kritik zu unserem Machine-Learning-Modell oder zu unserer Anwendung haben, würden wir uns sehr über Ihr Feedback freuen!")
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
    "Zum Modell": page1,
    "Hintergrund": page2,
    "Nächste Schritte": page3,
    "Feedback": page4
}

# Streamlit-App
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Gehe zu", list(pages.keys()))
    page = pages[selection]
    page()

if __name__ == "__main__":
    main()
