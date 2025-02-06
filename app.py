import pickle
import numpy as np
import streamlit as st
from keras.models import load_model

# Set page title and favicon
st.set_page_config(page_title="Lung Cancer Prediction", page_icon=":hospital:")

# Load the trained model and scaler
@st.cache_resource  # Cache the model and scaler for better performance
def load_assets():
    model = load_model('my_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# Title of the app
st.title("Lung Cancer Risk Prediction App")

# Sidebar for user input
st.sidebar.header("User Input Features")

# Function to collect user input
def get_user_input():
    age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=30)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    air_pollution = st.sidebar.slider("Air Pollution", 0, 10, 5)
    alcohol_use = st.sidebar.slider("Alcohol Use", 0, 10, 5)
    dust_allergy = st.sidebar.slider("Dust Allergy", 0, 10, 5)
    occupational_hazards = st.sidebar.slider("Occupational Hazards", 0, 10, 5)
    genetic_risk = st.sidebar.slider("Genetic Risk", 0, 10, 5)
    chronic_lung_disease = st.sidebar.slider("Chronic Lung Disease", 0, 10, 5)
    balanced_diet = st.sidebar.slider("Balanced Diet", 0, 10, 5)
    obesity = st.sidebar.slider("Obesity", 0, 10, 5)
    smoking = st.sidebar.slider("Smoking", 0, 10, 5)
    passive_smoker = st.sidebar.slider("Passive Smoker", 0, 10, 5)
    chest_pain = st.sidebar.slider("Chest Pain", 0, 10, 5)
    coughing_blood = st.sidebar.slider("Coughing of Blood", 0, 10, 5)
    fatigue = st.sidebar.slider("Fatigue", 0, 10, 5)
    weight_loss = st.sidebar.slider("Weight Loss", 0, 10, 5)
    shortness_of_breath = st.sidebar.slider("Shortness of Breath", 0, 10, 5)
    wheezing = st.sidebar.slider("Wheezing", 0, 10, 5)
    swallowing_difficulty = st.sidebar.slider("Swallowing Difficulty", 0, 10, 5)
    clubbing = st.sidebar.slider("Clubbing of Finger Nails", 0, 10, 5)
    frequent_cold = st.sidebar.slider("Frequent Cold", 0, 10, 5)
    dry_cough = st.sidebar.slider("Dry Cough", 0, 10, 5)
    snoring = st.sidebar.slider("Snoring", 0, 10, 5)

    # Encode gender
    gender = 1 if gender == "Male" else 2

    # Create a numpy array to hold the input data
    input_data = np.array([[
        age, gender, air_pollution, alcohol_use, dust_allergy, occupational_hazards,
        genetic_risk, chronic_lung_disease, balanced_diet, obesity, smoking,
        passive_smoker, chest_pain, coughing_blood, fatigue, weight_loss,
        shortness_of_breath, wheezing, swallowing_difficulty, clubbing,
        frequent_cold, dry_cough, snoring
    ]])

    return input_data

# Collect user input
input_data = get_user_input()

# Display user input
st.subheader("User Input Features")
st.write(input_data)

# Standardize the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
predicted_class = np.argmax(prediction, axis=1)[0]

# Map prediction to outcome
if predicted_class == 0:
    outcome = "Low"
elif predicted_class == 1:
    outcome = "Medium"
else:
    outcome = "High"

# Display results
st.subheader("Prediction")
st.write(f"The predicted risk level is: **{outcome}**")