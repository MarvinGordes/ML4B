# Notwendige Bibliotheken importieren
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import csv
import random
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import tsfresh
from io import StringIO
import pickle
import streamlit as st
from PIL import Image

liveSamples = []

st.title("Audio-Classification")



uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    file = pd.read_csv(uploaded_file)
    st.write(file)
    
    file.drop(['time'], axis=1, inplace=True) 
    file['dBFS'] = file['dBFS'].round(decimals = 0)
    file['Label'] = 'Knocking'
    
    liveSamples.append(file)  
    
    
     
    st.write("Dein Sample: ")

    df= file
    fig, ax = plt.subplots()
    ax.plot(df['seconds_elapsed'], df['dBFS'])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Liniengraph')
    st.pyplot(fig)



pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)



input = []


def prediction(input):  
    for i in liveSamples:
        input.append(pd.DataFrame({"dBFS_Varianz": i["dBFS"].var(),"dBFS_STD" : i["dBFS"].std(), "dBFS_mean" : i["dBFS"].mean(),
        "dBFS_min" : i["dBFS"].min(), "dBFS_max" : i["dBFS"].max(), "dBFS_absMax" : i["dBFS"].abs().max(), "dBFS_sum" : i["dBFS"].sum(), "dBFS_median" : i["dBFS"].median(), "Label" : i["Label"]}))
    input[0] = input[0].drop('Label', axis=1)
    prediction = classifier.predict(input)
    print(prediction)
    return prediction

if st.button("Predict"):
    prediction(input)
