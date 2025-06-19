from data_loader import load_stock_data
from preprocessing import scale_data, create_sequences
from model import build_lstm_model
from predict_and_evaluate import evaluate_model
from plotting import plot_predictions

import numpy as np
from sklearn.model_selection import train_test_split

# === 1. Load data ===
ticker = 'AAPL'
start_date = '2018-01-01'
end_date = '2024-12-31'

data = load_stock_data(ticker, start_date, end_date)
prices = data['Price'].values

# === 2. Preprocess ===
scaled_prices, scaler = scale_data(prices)
lookback = 30

X, y = create_sequences(scaled_prices, lookback)
X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

# === 3. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# === 4. Build and train model ===
model = build_lstm_model(input_shape=(lookback, 1))
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# === 5. Evaluate ===
results = evaluate_model(model, X_test, y_test, scaler)

print(f"MSE: {results['mse']:.4f}")
print(f"MAE: {results['mae']:.4f}")

# === 6. Plot predictions ===
plot_predictions(
    actual=results['actual'],
    predicted=results['predictions'],
    title=f"LSTM Prediction on {ticker}",
    save_path=f"plots/{ticker.lower()}_lstm_prediction.png"
)
