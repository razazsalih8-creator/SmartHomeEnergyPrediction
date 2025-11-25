import streamlit as st
import joblib
import numpy as np

# Load model & scaler
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("🏠 Smart Home Energy Prediction Dashboard")
st.write("Enter environmental sensor values to predict appliance energy consumption.")

feature_names = [
    'lights','T1','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5',
    'T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','T_out','Press_mm_hg',
    'RH_out','Windspeed','Visibility','Tdewpoint','hour','day_of_week',
    'month','week_of_year'
]

input_data = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    input_data.append(value)

if st.button("Predict Energy Consumption"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Appliance Energy Use: **{prediction:.2f} Wh**")
