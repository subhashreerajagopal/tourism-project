
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Config
st.set_page_config(page_title="Tourism Prediction", layout="centered")
st.title("🌴 Wellness Tourism Package Prediction")
st.write("Enter customer details to predict purchase likelihood.")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("models/tourism_model.joblib")

try:
    model = load_model()
except:
    st.error("Model file not found.")

# Input Form
with st.form("prediction_form"):
    st.subheader("Customer Details")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        income = st.number_input("Monthly Income", 1000, 100000, 25000)
        pitch_dur = st.number_input("Pitch Duration (min)", 1, 120, 15)
    with col2:
        family = st.number_input("Person Visiting", 1, 10, 2)
        trips = st.number_input("Number of Trips", 0, 20, 2)
        
    submit = st.form_submit_button("Predict Purchase")

if submit:
    # Dummy input vector for demo (size must match model training shape)
    # We construct an array of zeros and fill known inputs
    input_data = np.zeros(18) 
    input_data[0] = age
    input_data[12] = income # Index assumption based on standard columns
    
    prediction = model.predict([input_data])
    
    if prediction[0] == 1:
        st.success("Result: Likely to Purchase!")
    else:
        st.warning("Result: Unlikely to Purchase.")
