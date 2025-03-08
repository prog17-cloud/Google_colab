import streamlit as st

import numpy as np


# Load the model
import tensorflow as tf
import os

filename = 'Stock Predictions Model.keras'

# Check in the current directory
if os.path.exists(filename):
    filepath = filename
else:
    #Example of checking a subdirectory called models.
    filepath = os.path.join("models", filename)
    if not os.path.exists(filepath):
      filepath = None

if filepath:
    try:
        model = tf.keras.models.load_model(filepath)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"File not found: {filename}")

st.title('Stock Prediction App')

# Get input from the user
input_data = st.text_input('Enter input data (comma-separated numbers):')

if st.button('Predict'):
    if input_data:
        try:
            # Validate and preprocess input
            values = input_data.split(',')
            float_values = []
            for val in values:
                try:
                    float_values.append(float(val.strip())) # .strip() removes whitespace
                except ValueError:
                    st.error(f"Invalid input: '{val.strip()}' is not a number.")
                    st.stop()  # Stop if invalid input is found

            input_array = np.array(float_values).reshape(1, -1)

            # Make prediction
            prediction = model.predict(input_array)

            # Display prediction
            st.write('Prediction:', prediction)

        except Exception as e:
            st.error(f'An unexpected error occurred: {e}')
    else:
        st.warning('Please enter input data.')


