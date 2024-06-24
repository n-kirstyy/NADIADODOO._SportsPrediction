#####
#
#Disclaimer: This code was written with the help of Chat GPT and Copilot
#
####

import streamlit as st 
import pandas as pd
import numpy as np
import joblib
import os
import sklearn
import sklearn.metrics as metrics 



# This function is used to make a prediction
def predict_rating(model, input_features):
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

def main():
    model = joblib.load('random_forest_model.pkl')
    st.title('Football Rating Prediction')
    st.write('Adjust the values of the 15 features below as necessary to find the overall rating')
    
    st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffd1df;
        color: #333333;
    }
    .stButton > button {
        background-color: #B24BB2;
        color: white;
    }
    
    """,
    unsafe_allow_html=True
    )

    # Input features
    movement_reactions = st.number_input('Player Reactions Attribute', min_value=20, max_value=96, value=50, key='movement_reactions')
    mentality_composure = st.number_input('Player Mentality Composure', min_value=0, max_value=100, value=50, key='mentality_composure')
    potential = st.number_input('Player Potential Overall Attibute', min_value=0, max_value=100, value=50, key='potential')
    lf = st.number_input('Player Attribute Playing as LF', min_value=0, max_value=100, value=50, key='lf')
    cf = st.number_input('Player Attribute Playing as CF', min_value=0, max_value=100, value=50, key='cf')
    rf = st.number_input('Player Attribute Playing RF', min_value=0, max_value=100, value=50, key='rf')
    wage_eur = st.number_input('Player Weekly Wage (in EUR)', min_value=0, value=0, key='wage_eur')
    lw = st.number_input('Player Attribute Playing as LW', min_value=0, max_value=100, value=50, key='lw')
    rw = st.number_input('Player Attribute Playing RW', min_value=0, max_value=100, value=50, key='rw')
    power_shot_power = st.number_input('Player Power Shot Power', min_value=0, max_value=100, value=50, key='power_shot_power')
    lwb = st.number_input('Player Attribute Playing as LWB', min_value=0, max_value=100, value=50, key='lwb')
    rwb = st.number_input('Player Attribute Playing as RWB', min_value=0, max_value=100, value=50, key='rwb')
    value_eur = st.number_input('Player Value ( in EUR)', min_value=0, value=0, key='value_eur')
    release_clause_eur = st.number_input('Player Release Clause (in EUR)', min_value=374, value=5000, key='release_clause_eur')
    passing = st.number_input('Player Passing Attribute', min_value=20, max_value=93, value=50, key='passing')

    if st.button('Rating'):
        input_features = [
            movement_reactions, mentality_composure, potential, lf, cf, rf,
            wage_eur, lw, rw, power_shot_power, lwb, rwb, value_eur, release_clause_eur, passing
        ]
        try:
            rf_prediction = predict_rating(model, input_features)
            st.success(f'Random Forest predicted overall rating: {rf_prediction:.2f}')
        except Exception as e:
            st.error(f'Error occured during prediction: {e}')

if __name__ == '__main__':
    main()

