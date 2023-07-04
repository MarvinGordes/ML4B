import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def load(type,setSize):
    dataset = []
    path = 'input/'
    for x in range(1, setSize+1):
        dataset.append(path + type + str(x) + '.csv') 
    return dataset

def prep(dataset):
    data = []
    for x in range(len(dataset)):
        df = pd.read_csv(dataset[x])
        df = df.drop('time', axis=1)
        df = df.query("seconds_elapsed <= 5.0")
        data.append(df)
    return data

def plot(data):
    for x in range(len(data)):
        fig, ax = plt.subplots()
        ax.plot(data[x]["seconds_elapsed"], data[x]["dBFS"])
        ax.set_title("Datensatz " + str(x+1))
        st.pyplot(fig)
        
def plot2(data):
    num_cols = 2
    num_plots = len(data)
    num_rows = (num_plots + num_cols - 1) // num_cols
    plot_idx = 0
    for i in range(num_rows):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            if plot_idx < num_plots:
                with cols[j]:
                    fig, ax = plt.subplots()
                    ax.plot(data[plot_idx]['seconds_elapsed'], data[plot_idx]['dBFS'])
                    ax.set_title("Datensatz " + str(plot_idx))
                    st.pyplot(fig)
                plot_idx += 1
 
 
       
setSize = 10
plot2(prep(load('knock', setSize)))