
import streamlit as st
import tensorflow as tf

import numpy as np

# Load your model (adjust path if needed)
import tensorflow as tf
import os

filepath = "Stock Predictions Model.keras"  # Replace with the actual path
if os.path.exists(filepath):
    try:
        model = tf.keras.models.load_model(filepath)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"File not found: {filepath}")
  


st.title('Stock Prediction App')

# Get input from the user
input_data = st.text_input('Enter input data (e.g., comma-separated values):')

if st.button('Predict'):
    if input_data:
        try:
            # Preprocess input (replace with your actual preprocessing)
            input_array = np.array([float(x) for x in input_data.split(',')])
            input_array = input_array.reshape(1, -1)  # Reshape for model input

            # Make prediction
            prediction = model.predict(input_array)

            # Display prediction
            st.write('Prediction:', prediction)
        except Exception as e:
            st.error(f'Error: {e}')
    else:
        st.warning('Please enter input data.')



