import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

def run():
    #Membuat judul
    st.title('Water Quality Predict')

    #membuat batas dengan garis lurus
    st.markdown('---')

    #Membuat sub header
    st.subheader('Exploratory Data Analysis for Water Quality Predict')

    #Menambahkan gambar
    image = Image.open('water.jpg')
    st.image(image, caption = 'Water Quality')

    #Menambahkan deskripsi
    st.write('Page ini dibuat oleh Raodah Hasman - RMT 030')

    #Menambahkan deskripsi
    st.write('Objective : Project ini dibuat untuk meningkatkan kemampuan model dalam memprediksi air apakah tergolong aman atau tidak sehingga air dapat digunakan/dimanfaatkan sesuai dengan kebutuhan masyarakat.')

    #membuat batas dengan garis lurus
    st.markdown('---')

    #Show dataframe
    df = pd.read_csv('waterQuality1.csv')
    st.dataframe(df)

    # Membuat bar plot fitur variabel target
    st.write('## Distribusi Is Safe')
    fig = plt.figure(figsize=(15, 6))
    sns.countplot(x='is_safe', data = df)
    st.pyplot(fig)

    # Menapilkan distribusi fitur Barium
    st.write('## Distribusi Barium')
    fig = plt.figure(figsize=(15, 6))
    sns.histplot(df['barium'], kde=True, bins=30)
    plt.title(f'Histogram of Barium')
    plt.xlabel('Barium')
    plt.ylabel('Frequency')
    st.pyplot(fig)

    #Menampilkan distribusi berdasarkan input user
    st.write('## Distribusi berdasarkan input user')
    option = st.selectbox('pilih column : ', ('nitrites', 'aluminium', 'arsenic', 'cadmium', 'bacteria', 'copper', 'silver', 'uranium'))
    fig = plt.figure(figsize=(15, 6))
    sns.histplot(df[option], bins = 30, kde = True)
    st.pyplot(fig)

    #Menampilkan boxplot fitur Aluminium
    st.write('## Boxplot Aluminium')
    fig = plt.figure(figsize=(15, 6))
    sns.boxplot(df['aluminium'])
    plt.title('Boxplot of Aluminium')
    plt.xlabel('Aluminium')
    plt.ylabel('Count')
    st.pyplot(fig)


if __name__ == '__main__':
    run()