import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('BTC-5m-500-OHLCV.csv')

# Create DataFrame
df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume"])
df["open_time"] = pd.to_datetime(df["open_time"])

# Feature engineering
df["time_diff"] = df["open_time"].diff().dt.total_seconds().fillna(0)
df["price_diff"] = df["close"].diff().fillna(0)

# Features and target
features = df[["open", "high", "low", "close", "volume", "time_diff", "price_diff"]]
target = df["close"].shift(-1).ffill()  # Use .ffill() to fill NaN values

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define and train the model with increased bounds
kernel = C(1.0, (1e-4, 1e2)) * RBF(1, (1e-4, 1e10))  # Increase upper bound for length_scale to 1e3
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
gpr.fit(X_train, y_train)

# Predictions
y_pred, sigma = gpr.predict(X_test, return_std=True)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Predict next candle close price
next_features = features.iloc[-1:].copy()  # Create a DataFrame with the same feature names
next_close_pred, next_sigma = gpr.predict(next_features, return_std=True)
print(f"Next Candle Close Price Prediction: {next_close_pred[0]} ± {next_sigma[0]}")

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Values')
plt.plot(y_pred, label='Predicted Values')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.title('Actual vs. Predicted Close Prices')
plt.show()

# Plot residuals
residuals = y_test.values - y_pred
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.title('Residuals (Actual - Predicted)')
plt.show()

# Baseline model for comparison (predicting the previous close price)
baseline_pred = X_test['close'].shift(1).fillna(X_train['close'].iloc[-1])
baseline_mse = mean_squared_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(baseline_mse)
print(f"Baseline Model MSE: {baseline_mse}")
print(f"Baseline Model RMSE: {baseline_rmse}")