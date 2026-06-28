from src.predict import predict_with_probability

import streamlit as st
import pandas as pd

CHEST_PAIN = ["asymptomatic", "non-anginal", "atypical angina", "typical angina"]
ECG_RESULT = ["normal", "lv hypertrophy", "st-t abnormality"]
SLOPE = ["upsloping", "flat", "downsloping"]
THAL = ["normal", "reversable defect", "fixed defect"]

st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀", layout="wide")
st.title("🫀 Heart Disease Prediction")
st.write("Provide the patient information below to predict the likelihood of the heart disease")

with st.container(border=True):
    st.subheader("📋 Patient Information")

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            age = st.number_input("Age", min_value=1, max_value=100, value=30)
            chest_pain = st.selectbox("Chest Pain", CHEST_PAIN)
            serum_chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=80, max_value=200)
            ecg_result = st.selectbox("Resting ECG Results", ECG_RESULT)
            exer_ind_angina = st.selectbox("Exercise Induced Angina", [False, True])
            slope = st.selectbox("Slope", SLOPE)
            thal = st.selectbox("Thal", THAL)

        with right:
            sex = st.selectbox("Sex", ["Male", "Female"])
            rest_bp = st.number_input("Resting BP", min_value=60, max_value=200)
            fast_blood_sugar = st.selectbox("Fasting Blood Sugar", [False, True])
            heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=200)
            oldpeak = st.number_input("Old Peak", min_value=0.0, max_value=10.0, step=0.1)
            major_vessels = st.number_input("No. of major vessels", min_value=0, max_value=3, step=1)


        col1, col2, col3 = st.columns([2, 1, 2])

        with col2:
            predict_btn = st.form_submit_button(
                "Predict",
                use_container_width=True
            )

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

    st.divider()

    with st.container(border=True):

        st.subheader("🩺 Prediction Result")

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

        left, right = st.columns([2, 1])

        with left:
            st.metric("Prediction Probability", f"{probability:.2%}")

        with right:
            if prediction == 1:
                st.metric("Risk Level", "🔴 High")
            else:
                st.metric("Risk Level", "🟢 Low")


st.divider()
st.subheader("ℹ️ Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Linear SVM")
with col2:
    st.metric("Accuracy", "86.89%")

with col3:
    st.metric("Recall", "86.21%")

st.caption(
    """
    This application predicts the likelihood of heart disease using a
    Support Vector Machine (SVM) trained on the UCI Cleveland Heart Disease Dataset.
    The model performs preprocessing (missing value imputation, feature
    scaling, and one-hot encoding) automatically before making predictions.
    """
)