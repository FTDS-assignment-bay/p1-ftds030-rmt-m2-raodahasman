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
        aluminium = st.number_input('aluminium', min_value=0.0, value=1.0)
        ammonia = st.number_input('ammonia', min_value=0.0, value=0.9)
        arsenic = st.number_input('arsenic', min_value=0.0, value=0.5)
        barium = st.number_input('barium', min_value=0.0, value=0.0)
        cadmium = st.number_input('cadmium', min_value=0.0, value=0.1)
        chloramine = st.number_input('cloromine', min_value=0.0, value=0.1)
        chromium = st.number_input('chromium', min_value=0.0, value=0.0)
        copper = st.number_input('copper', min_value=0.0, value=0.7)
        flouride = st.number_input('flouride', min_value=0.0, value=0.6)
        bacteria = st.number_input('bacteria', min_value=0.0, value=0.4)
        viruses = st.number_input('viruses', min_value=0.0, value=0.2)
        lead = st.number_input('lead', min_value=0.0, value=0.9)
        nitrates = st.number_input('nitrites', min_value=0.0, value=0.3)
        nitrites = st.number_input('nitrites', min_value=0.0, value=0.0)
        mercury = st.number_input('mercury', min_value=0.0, value=0.1)
        perchlorate = st.number_input('perchlorate', min_value=0.0, value=0.2)
        radium = st.number_input('radium', min_value=0.0, value=0.5)
        selenium = st.number_input('selenium', min_value=0.0, value=0.2)
        silver = st.number_input('silver', min_value=0.0, value=0.0)
        uranium = st.number_input('uranium', min_value=0.0, value=0.1)

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


