from src.predict import predict_with_probability

import streamlit as st
import pandas as pd

st.title("Heart Disease Prediction")
st.write("Provide the patient information below to predict the likelihood of the heart disease")

left, right = st.columns(2)

with left:
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    chest_pain = st.selectbox(
        "Chest Pain", 
        ["asymptomatic", "non-anginal", "atypical angina", "typical angina"]
    )
    serum_chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=80, max_value=200)
    ecg_result = st.selectbox("Resting ECG Results", ["normal", "lv hypertrophy", "st-t abnormality"])
    exer_ind_angina = st.selectbox("Exercise Induced Angina", [False, True])
    slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping"])
    thal = st.selectbox("Thal", ["normal", "reversable defect", "fixed defect"])

with right:
    sex = st.selectbox("Sex", ["Male", "Female"])
    rest_bp = st.number_input("Resting BP", min_value=60, max_value=200)
    fast_blood_sugar = st.selectbox("Fasting Blood Sugar", [False, True])
    heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=200)
    oldpeak = st.number_input("Old Peak", min_value=0.0, max_value=10.0, step=0.1)
    major_vessels = st.number_input("No. of major vessels", min_value=0, max_value=3, step=1)


predict_btn = st.button("Predict")

if predict_btn:
    patient = {
        "Age": age,
        "Sex": sex,
        "Chest Pain": chest_pain,
        "Resting BP": rest_bp,
        "Serum Cholestrol(mg/dl)": serum_chol,
        "Fasting Blood Sugar (> 120 mg/dl)": fast_blood_sugar,
        "Resting ECG Result": ecg_result,
        "Max Heart Rate": heart_rate,
        "Angina (Exercise Induced)": exer_ind_angina,
        "Oldpeak": oldpeak,
        "Slope": slope,
        "No. of Major Vessels": major_vessels,
        "Thal": thal
    }

    patient_df = pd.DataFrame([patient])
    prediction, probability = predict_with_probability(patient_df)

    if prediction == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")

    st.write(f"Probability: {probability:.2%}")