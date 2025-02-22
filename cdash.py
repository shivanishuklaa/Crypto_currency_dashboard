import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Define supported cryptocurrencies
CRYPTO_OPTIONS = {
    "Ethereum (ETH)": "ETH-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Ripple (XRP)": "XRP-USD",
    "Solana (SOL)": "SOL-USD"
}

# Streamlit app title
st.title("📈 Cryptocurrency Dashboard with LSTM Prediction")

# User input: Select a cryptocurrency
crypto_name = st.selectbox("Select Cryptocurrency", list(CRYPTO_OPTIONS.keys()))

# User input: Select date range (2 years to 1 day)
date_range = st.slider("Select Time Period (in days)", 1, 1095, 180)  # Default 180 days

# Fetch historical data
@st.cache_data
def fetch_data(symbol, days):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    df = yf.download(symbol, start=start_date, end=end_date)

    if df.empty:
        st.error("No data fetched! Please try a different time range or cryptocurrency.")
        return None  

    # Flatten MultiIndex column names
    df.columns = df.columns.get_level_values(0)  
    df.reset_index(inplace=True)  
    
    return df

# Load data
symbol = CRYPTO_OPTIONS[crypto_name]
data = fetch_data(symbol, date_range)

if data is None:
    st.stop()  # Stop execution if no data is fetched

# Display raw data
st.subheader(f"{crypto_name} Price Data (Last {date_range} Days)")
st.write(data.tail())

# Price trend visualization
fig_price = px.line(data, x="Date", y="Close", title=f"{crypto_name} Price Trend", labels={"Close": "Price (USD)", "Date": "Date"})
st.plotly_chart(fig_price)

# Volume trend visualization
fig_volume = px.line(data, x="Date", y="Volume", title=f"{crypto_name} Trading Volume", labels={"Volume": "Volume", "Date": "Date"})
st.plotly_chart(fig_volume)

# Data Preprocessing for LSTM
def prepare_lstm_data(df, time_step=60):
    if len(df) < time_step:  # Ensure enough data points
        st.error("Not enough data for LSTM training! Try increasing the date range.")
        return None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']].values)

    X, Y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        Y.append(scaled_data[i, 0])

    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM

    return X, Y, scaler

# LSTM Model Training
@st.cache_resource
def train_lstm(df):
    X_train, Y_train, scaler = prepare_lstm_data(df)
    
    if X_train is None:
        return None, None, None, None  # Handle insufficient data case
    
    # Define LSTM Model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, Y_train, epochs=30, batch_size=32, verbose=0)
    
    return model, scaler, X_train, Y_train

# Train LSTM Model
st.subheader(f" {crypto_name} Price Prediction (Next 3-6 Months)")
model, scaler, X_train, Y_train = train_lstm(data)

# Ensure model training was successful
if model is None:
    st.error("Not enough data to train the model! Please increase the date range.")
    st.stop()

# Model Performance (MSE)
y_actual = scaler.transform(data[['Close']].values)[-len(Y_train):]
y_pred = model.predict(X_train, verbose=0)
mse = mean_squared_error(y_actual, y_pred)
# st.subheader(" Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.5f}")

# User Input for Forecast Period
st.subheader(" Price Prediction for the Next 3-6 Months")
n_months = st.slider("Months of prediction:", 3, 6)  # User selects 3-6 months
period = n_months * 30  # Convert months to days

# Predict Future Prices
def predict_future(model, df, scaler, days=180):
    last_60_days = df[['Close']].values[-60:]  
    last_60_days_scaled = scaler.transform(last_60_days)

    X_pred = []
    X_pred.append(last_60_days_scaled)
    X_pred = np.array(X_pred)
    X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))

    future_prices = []
    for _ in range(days):
        predicted_price = model.predict(X_pred, verbose=0)
        future_prices.append(predicted_price[0, 0])

        # Append new prediction and remove the first one (rolling window)
        X_pred = np.append(X_pred[:, 1:, :], np.reshape(predicted_price, (1, 1, 1)), axis=1)

    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
    
    future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, days + 1)]
    future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_prices.flatten()})

    return future_df

# Generate future predictions
future_df = predict_future(model, data, scaler, days=period)

# Combine Actual and Predicted Data
actual_df = data[['Date', 'Close']]
actual_df.rename(columns={'Close': 'Actual Price'}, inplace=True)

# Merge actual and predicted data
combined_df = pd.concat([actual_df, future_df], ignore_index=True)

# Plot Actual vs. Predicted Prices
fig_combined = px.line(combined_df, x="Date", y=["Actual Price", "Predicted Price"],
                        title=f"{crypto_name} Price Forecast vs. Actual",
                        labels={"value": "Price (USD)", "Date": "Date"},
                        color_discrete_map={"Actual Price": "blue", "Predicted Price": "red"})

st.plotly_chart(fig_combined)

st.success(" Dashboard Successfully Loaded!")
