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



st.title("Audio-Classification")

pickle_in = open('knnclassifier3inputs.pkl', 'rb')
classifier = pickle.load(pickle_in)


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    file = pd.read_csv(uploaded_file)
    st.write(file)
    
    file.drop(['time'], axis=1, inplace=True) 
    file['dBFS'] = file['dBFS'].round(decimals = 0)
    file['Label'] = 'Knocking'
     
    st.write("Dein Sample: ")

    df= file
    fig, ax = plt.subplots()
    ax.plot(df['seconds_elapsed'], df['dBFS'])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Liniengraph')
    st.pyplot(fig)






def prediction(input):
    data = []  
    data.append(pd.DataFrame({"dBFS_Varianz": input["dBFS"].var(),"dBFS_STD" : input["dBFS"].std(), "dBFS_mean" : input["dBFS"].mean(),
        "dBFS_min" : input["dBFS"].min(), "dBFS_max" : input["dBFS"].max(), "dBFS_absMax" : input["dBFS"].abs().max(), "dBFS_sum" : input["dBFS"].sum(), "dBFS_median" : input["dBFS"].median(), "Label" : input["Label"]}))
    
    
    data[0] = data[0].drop('Label', axis=1)
    prediction = classifier.predict(data[0])
    prediction = prediction[0]
    st.write(data[0].head(1))  
    return prediction

if st.button("Predict"):
    st.write(prediction(file))