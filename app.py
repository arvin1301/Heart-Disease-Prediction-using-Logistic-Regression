import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# Load the Saved Model and Scaler
# ===============================
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# Streamlit App Configuration
# ===============================
st.set_page_config(page_title="❤️ Heart Disease Predictor", layout="centered")
st.title("🩺 Framingham Heart Disease Prediction App")
st.write("Predict the 10-year risk of Coronary Heart Disease (CHD) using patient data.")

# ===============================
# Input Form
# ===============================
st.header("Enter Patient Details:")

col1, col2 = st.columns(2)

with col1:
    male = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    education = st.selectbox("Education Level (1–4)", [1, 2, 3, 4])
    currentSmoker = st.selectbox("Current Smoker (1 = Yes, 0 = No)", [1, 0])
    cigsPerDay = st.number_input("Cigarettes per Day", min_value=0, max_value=70, value=10)
    BPMeds = st.selectbox("On BP Medication (1 = Yes, 0 = No)", [1, 0])
    prevalentStroke = st.selectbox("Prevalent Stroke (1 = Yes, 0 = No)", [1, 0])
    prevalentHyp = st.selectbox("Prevalent Hypertension (1 = Yes, 0 = No)", [1, 0])

with col2:
    diabetes = st.selectbox("Diabetes (1 = Yes, 0 = No)", [1, 0])
    totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=700, value=230)
    sysBP = st.number_input("Systolic BP (mm Hg)", min_value=80, max_value=250, value=130)
    diaBP = st.number_input("Diastolic BP (mm Hg)", min_value=50, max_value=150, value=85)
    BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    heartRate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=75)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=40, max_value=400, value=85)

# ===============================
# Prediction Section
# ===============================
if st.button("🔍 Predict"):
    try:
        # Create input in the same order as training
        features = np.array([[male, age, education, currentSmoker, cigsPerDay, BPMeds,
                              prevalentStroke, prevalentHyp, diabetes, totChol,
                              sysBP, diaBP, BMI, heartRate, glucose]])

        # Scale the input
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        # Display result
        if prediction == 1:
            st.error(f"🚨 High risk of Heart Disease ({probability*100:.2f}%). Please consult your doctor.")
        else:
            st.success(f"✅ Low risk of Heart Disease ({probability*100:.2f}%). Keep maintaining a healthy lifestyle!")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("Developed by Arvind Sharma | Powered by Streamlit & Scikit-learn")
