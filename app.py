import streamlit as st
import joblib
import numpy as np

# Load saved model and encoders
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Streamlit UI
st.title("Employee Salary Class Prediction")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
education = st.selectbox("Education", encoders["education"].classes_)
occupation = st.selectbox("Occupation", encoders["occupation"].classes_)
hours_per_week = st.slider("Hours per week", min_value=1, max_value=100, value=40)

if st.button("Predict"):
    try:
        # Encode inputs
        education_encoded = encoders["education"].transform([education])[0]
        occupation_encoded = encoders["occupation"].transform([occupation])[0]

        # Combine inputs
        X_input = np.array([[age, education_encoded, occupation_encoded, hours_per_week]])
        X_scaled = scaler.transform(X_input)

        # Predict
        prediction = model.predict(X_scaled)[0]
        prediction_label = target_encoder.inverse_transform([prediction])[0]

        # Display result
        st.success(f"Predicted Salary Class: {prediction_label}")
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
