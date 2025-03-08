import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import joblib #add joblib

filepath = 'Stock Predictions Model.keras'
print(f"Filepath: {filepath}")
if os.path.exists(filepath):
    print("File exists.")
    try:
        model = tf.keras.models.load_model(filepath)
        scaler = joblib.load('scaler.joblib') #load the scaler.
        print("Model and scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        model = None
        scaler = None
else:
    print("File does not exist.")
    print(f"Current working directory: {os.getcwd()}")
    model = None
    scaler = None

# ... (rest of your Streamlit app code) ...

if model is not None and scaler is not None: #check if the model and scaler are loaded.
    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])

    x,y = np.array(x), np.array(y)

    try:
        predict = model.predict(x)

        scale = 1/scaler.scale_

        predict = predict * scale
        y = y * scale

        st.subheader('Original Price vs Predicted Price')
        fig4 = plt.figure(figsize=(8,6))
        plt.plot(predict, 'r', label='Predicted Price') #changed the label.
        plt.plot(y, 'g', label = 'Original Price') #changed the label.
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.show()
        st.pyplot(fig4)
    except Exception as e:
        st.error(f"Prediction error: {e}")
else:
    st.error("Model or scaler not loaded. Cannot make predictions.")

