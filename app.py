import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and preprocessing pipeline
aussie_rain = joblib.load('models/rf_model.joblib')
model = aussie_rain['model']
imputer = aussie_rain['imputer']
scaler = aussie_rain['scaler']
encoder = aussie_rain['encoder']
numeric_cols = aussie_rain['numeric_cols']
categorical_cols = aussie_rain['categorical_cols']
encoded_cols = aussie_rain['encoded_cols']
min_values = aussie_rain['min_values']
max_values = aussie_rain['max_values']
mean_values = aussie_rain['mean_values']


st.title("Rain Prediction in Australia")
st.image("images/zones.jpg", use_container_width = True)
st.markdown("**Enter today's weather data and see if it will rain 🌧️ tomorrow.**")

# --- User Inputs ---
user_input = {}

# Categorical inputs
# Get the categories from the encoder (same order as categorical_cols)
encoder_categories = encoder.categories_

# Mapping technical column names to user-friendly labels
label_map = {
    'Location': 'Location',
    'WindGustDir': 'Wind Gust Direction',
    'WindDir9am': 'Wind Direction at 9 AM',
    'WindDir3pm': 'Wind Direction at 3 PM',
    'RainToday': 'Did it rain today?',
    
    # Add friendly names for numeric columns too:
    'MinTemp': 'Minimum Temperature (°C)',
    'MaxTemp': 'Maximum Temperature (°C)',
    'Rainfall': 'Rainfall (mm)',
    'Evaporation': 'Evaporation (mm)',
    'Sunshine': 'Sunshine Hours',
    'WindGustSpeed': 'Wind Gust Speed (km/h)',
    'Humidity9am': 'Humidity at 9 AM (%)',
    'Humidity3pm': 'Humidity at 3 PM (%)',
    'Pressure9am': 'Pressure at 9 AM (hPa)',
    'Pressure3pm': 'Pressure at 3 PM (hPa)',
    'Cloud9am': 'Cloud Cover at 9 AM (0–8)',
    'Cloud3pm': 'Cloud Cover at 3 PM (0–8)',
    'Temp9am': 'Temperature at 9 AM (°C)',
    'Temp3pm': 'Temperature at 3 PM (°C)'
}

# Categorical inputs
for col, categories in zip(categorical_cols, encoder_categories):
    label = label_map.get(col, col)  # Fallback to the raw col name if not in the map

    if col == 'Location':
        user_input[col] = st.selectbox(
            label,
            options=['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree', 'Newcastle', 'NorahHead',
                     'NorfolkIsland', 'Penrith', 'Richmond', 'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
                     'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat', 'Bendigo', 'Sale',
                     'MelbourneAirport', 'Melbourne', 'Mildura', 'Nhil', 'Portland', 'Watsonia', 'Dartmoor',
                     'Brisbane', 'Cairns', 'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
                     'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth', 'SalmonGums',
                     'Walpole', 'Hobart', 'Launceston', 'AliceSprings', 'Darwin', 'Katherine', 'Uluru']
        )
    else:
        # Use selectbox with a friendly label
        user_input[col] = st.selectbox(label, options=list(categories))

# Numeric inputs
for col in numeric_cols:
    label = label_map.get(col, col)
    user_input[col] = st.slider(
        label, 
        min_value=float(min_values[col]),
        max_value=float(max_values[col]),
        value=float(mean_values[col]),
        step=0.1
    )

# Predict button
if st.button("PREDICT RAIN TOMORROW"):
    # Create DataFrame from input
    input_df = pd.DataFrame([user_input])
    
    # Preprocess numeric features
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # Preprocess categorical features
    encoded_input = encoder.transform(input_df[categorical_cols])
    encoded_input_df = pd.DataFrame(encoded_input, columns=encoded_cols)
    
    # Combine all features
    final_input = pd.concat([input_df[numeric_cols], encoded_input_df], axis=1)
    
    # Predict
    prediction = model.predict(final_input)[0]
    proba_all = model.predict_proba(final_input)[0]
    classes = model.classes_  # ['No', 'Yes']

    # Get probability for the predicted class
    predicted_class_index = list(classes).index(prediction)
    predicted_class_proba = proba_all[predicted_class_index]

    # Choose emoji and message based on prediction
    if prediction == "Yes":
        headline = "🌧️ Rain is likely tomorrow!"    
    else:
        headline = "☀️ No rain expected tomorrow."

    # Display result
    st.subheader(headline)
    st.write(f"Model confidence: {predicted_class_proba:.2%}")
    
    