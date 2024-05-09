import streamlit as st
import pandas as pd
import numpy as np
import pickle

#load model
with open('model_dt.pkl', 'rb') as file:
  load_model = pickle.load(file)


def run():
  
    with st.form('form_water_quality'):
        st.write('Masukkan data untuk prediksi')
        aluminium = st.number_input('aluminium', value = 0, min_value = 0)
        ammonia = st.number_input('ammonia', value = 2, min_value = 0)
        arsenic = st.number_input('arsenic', value = 0, min_value = 0)
        barium = st.number_input('barium', value = 1, min_value = 0)
        cadmium = st.number_input('cadmium', value = 0, min_value = 0)
        chloramine = st.number_input('chloramine', value = 1, min_value = 0)
        chromium = st.number_input('chromium', value = 3, min_value = 0)
        copper = st.number_input('copper', value = 0, min_value = 0)
        flouride = st.number_input('flouride', value = 4, min_value = 0)
        bacteria = st.number_input('bacteria', value = 1, min_value = 0)
        viruses = st.number_input('viruses', value = 3, min_value = 0)
        lead = st.number_input('lead', value = 1, min_value = 0)
        nitrates = st.number_input('nitrites', value = 0, min_value = 0)
        nitrites = st.number_input('nitrites', value = 2, min_value = 0)
        mercury = st.number_input('mercury', value = 1, min_value = 0)
        perchlorate = st.number_input('perchlorate', value = 0, min_value = 0)
        radium = st.number_input('radium', value = 0, min_value = 0)
        selenium = st.number_input('selenium', value = 0, min_value = 0)
        silver = st.number_input('silver', value = 0)
        uranium = st.number_input('uranium', value = 0)

        #submit button form
        submitted = st.form_submit_button('Predict')

    data_inf = {
        'aluminium' : aluminium,
        'ammonia' : ammonia,
        'arsenic' : arsenic,
        'barium' : barium,
        'cadmium' : cadmium,
        'chloramine' : chloramine,
        'chromium' : chromium,
        'copper' : copper,
        'flouride' : flouride,
        'bacteria' : bacteria,
        'viruses' : viruses,
        'lead' : lead,
        'nitrates' : nitrates,
        'nitrites' : nitrites,
        'mercury' : mercury,
        'perchlorate' : perchlorate,
        'radium' : radium,
        'selenium' : selenium,
        'silver' : silver,
        'uranium' : uranium
    }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
   
        y_pred_dt = load_model.predict(data_inf)

        st.write('## Hasil prediksi:', y_pred_dt)


if __name__ == '__main__':
   run()


