import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import joblib  # Import joblib for loading the scaler

# Load the model and scaler
filepath = 'Stock Predictions Model.keras'
print(f"Filepath: {filepath}")
if os.path.exists(filepath):
    print("File exists.")
    try:
        model = tf.keras.models.load_model(filepath)
        scaler = joblib.load('scaler.joblib')  # Load the scaler
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

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.transform(data_test) #Use transform not fit_transform

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

if model is not None and scaler is not None:
    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i - 100:i])
        y.append(data_test_scale[i, 0])

    x, y = np.array(x), np.array(y)

    try:
        predict = model.predict(x)

        scale = 1 / scaler.scale_

        predict = predict * scale
        y = y * scale

        st.subheader('Original Price vs Predicted Price')
        fig4 = plt.figure(figsize=(8, 6))
        plt.plot(y, 'g', label='Original Price') #fixed labels
        plt.plot(predict, 'r', label='Predicted Price') #fixed labels
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        st.pyplot(fig4)
    except Exception as e:
        st.error(f"Prediction error: {e}")
else:
    st.error("Model or scaler not loaded. Cannot make predictions.")
