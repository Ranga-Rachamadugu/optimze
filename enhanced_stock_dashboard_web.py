
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 Enhanced Stock Price Prediction Dashboard")

# Sidebar
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))
models_to_run = st.sidebar.multiselect("Models", ["LSTM", "Random Forest"], default=["LSTM", "Random Forest"])
run = st.sidebar.button("Run Prediction")

if run:
    df = yf.download(ticker, start=start_date, end=end_date)
    df['Date'] = df.index
    df = df[['Date', 'Close']].reset_index(drop=True)

    # Prepare data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])
    window = 60
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    lstm_pred, rf_pred = None, None

    if "LSTM" in models_to_run:
        # LSTM
        lstm = Sequential()
        lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        lstm.add(Dropout(0.2))
        lstm.add(LSTM(50))
        lstm.add(Dropout(0.2))
        lstm.add(Dense(1))
        lstm.compile(optimizer='adam', loss='mean_squared_error')
        lstm.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        lstm_pred = scaler.inverse_transform(lstm.predict(X_test).reshape(-1, 1)).flatten()

    if "Random Forest" in models_to_run:
        # RF
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]), y_train)
        rf_pred = scaler.inverse_transform(rf.predict(X_test.reshape(X_test.shape[0], X_test.shape[1])).reshape(-1, 1)).flatten()

    # Actual
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    dates = df['Date'].iloc[-len(y_test):]

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_true, mode='lines', name='Actual'))
    if lstm_pred is not None:
        fig.add_trace(go.Scatter(x=dates, y=lstm_pred, mode='lines', name='LSTM'))
    if rf_pred is not None:
        fig.add_trace(go.Scatter(x=dates, y=rf_pred, mode='lines', name='RF'))
    fig.update_layout(title=f"{ticker} Stock Prediction", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    st.subheader("📊 Model Performance")
    if lstm_pred is not None:
        lstm_rmse = np.sqrt(mean_squared_error(y_true, lstm_pred))
        lstm_mae = mean_absolute_error(y_true, lstm_pred)
        st.markdown(f"**LSTM RMSE:** {lstm_rmse:.2f} | **MAE:** {lstm_mae:.2f}")
    if rf_pred is not None:
        rf_rmse = np.sqrt(mean_squared_error(y_true, rf_pred))
        rf_mae = mean_absolute_error(y_true, rf_pred)
        st.markdown(f"**RF RMSE:** {rf_rmse:.2f} | **MAE:** {rf_mae:.2f}")

    # Download predictions
    results = pd.DataFrame({"Date": dates, "Actual": y_true})
    if lstm_pred is not None:
        results["LSTM_Predicted"] = lstm_pred
    if rf_pred is not None:
        results["RF_Predicted"] = rf_pred

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions as CSV", data=csv, file_name=f"{ticker}_predictions.csv", mime="text/csv")
