import streamlit as st
import pandas as pd
import joblib

# Load the pipeline
pipeline = joblib.load('obesity_pipeline.pkl')

st.title("🍔 Obesity Risk Predictor")

# Input form
st.subheader("Enter Your Details")
age = st.slider("Age", 1, 100, 25)
gender = st.radio("Gender", ["Male", "Female"])
Height = st.slider("Height (m)", 0.5, 2.5, 2.0)
Weight = st.slider("Weight (kg)", 20, 200, 70)
family_history = st.selectbox("Family History of Obesity", ["Yes", "No"])
favc = st.selectbox("Frequent consumption of high caloric food (FAVC)", ["Yes", "No"])
fcvc = st.selectbox("Frequency of consumption of vegetables (FCVC)", [1, 2, 3])
caec = st.selectbox("Consumption of food between meals (CAEC)", ['Sometimes', 'No', 'Always', 'Frequently'])
ch2o = st.selectbox("Consumption of water daily (CH20)", [1, 2, 3])
smoke = st.selectbox("Smoke", ["Yes", "No"])
calc = st.selectbox("Consumption of alcohol (CALC)", ['Sometimes', 'No', 'Always', 'Frequently'])
scc = st.selectbox("Calories consumption monitoring (SCC)", ["Yes", "No"])
faf = st.selectbox("Physical activity frequency (FAF)", [0, 1, 2, 3])
tue = st.selectbox("Time using technology devices (TUE)", [0, 1, 2, 3])
mtrans = st.selectbox("Transportation used (MTRANS)", ['Automobile', 'Public_Transportation', 'Walking', 'Motorbike', 'Bike'])
ncp = st.selectbox("Number of main meals (NCP)", [1, 2, 3])

# Encoding mappings based on your training data
caec_map = {'No': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
calc_map = {'No': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
mtrans_map = {
    'Automobile': 0,
    'Public_Transportation': 1,
    'Walking': 2,
    'Motorbike': 3,
    'Bike': 4
}

input_dict = {
    'Age': age,
    'Gender': 1 if gender == "Male" else 0,
    'family_history_with_overweight': 1 if family_history == "Yes" else 0,
    'FAVC': 1 if favc == "Yes" else 0,
    'FCVC': fcvc,
    'NCP': ncp,
    'CAEC': caec_map.get(caec, 0),
    'CH2O': ch2o,
    'SMOKE': 1 if smoke == "Yes" else 0,
    'SCC': 1 if scc == "Yes" else 0,
    'FAF': faf,
    'TUE': tue,
    'CALC': calc_map.get(calc, 0),
    'MTRANS': mtrans_map.get(mtrans, 0),
    'Height': Height,
    'Weight': Weight,
}
input_df = pd.DataFrame([input_dict])

if st.button("Predict Obesity Risk"):
    prediction_encoded = pipeline.predict(input_df)[0]
    obesity_mapping = {
        1: 'Normal_Weight',
        3: 'Obesity_Type_I',
        4: 'Obesity_Type_II',
        6: 'Obesity_Type_III',
        5: 'Overweight_Level_I',
        0: 'Insufficient_Weight',
        2: 'Overweight_Level_II'
    }
    prediction = obesity_mapping.get(prediction_encoded, 'Unknown')
    st.success(f"🎯 Predicted Obesity Category: **{prediction}**")
