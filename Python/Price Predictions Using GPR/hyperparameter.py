import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import rsi
from ta.trend import MACD
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange
import os

# File path
script_dir = os.path.dirname(os.path.realpath(__file__))
read_data_file_path = os.path.join(script_dir, 'BTC-5m-500-OHLCV.csv')

# Read data
data = pd.read_csv(read_data_file_path)

# Create DataFrame
df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume"])
df["open_time"] = pd.to_datetime(df["open_time"])

# Feature engineering
df["time_diff"] = df["open_time"].diff().dt.total_seconds().fillna(0)
df["price_diff"] = df["close"].diff().fillna(0)

# Calculate RSI
df['rsi'] = rsi(close=df['close'], window=14)

# Calculate MACD
macd = MACD(df['close']).macd()
df['macd'] = macd

# Calculate Stochastic RSI
stoch_rsi = StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()
df['stoch_rsi'] = stoch_rsi

# Calculate Bollinger Bands
bb = BollingerBands(df['close'])
df['bb_upperband'] = bb.bollinger_hband()
df['bb_lowerband'] = bb.bollinger_lband()

# Calculate ATR
atr = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
df['atr'] = atr

# Features and target
features = df[["open", "high", "low", "close", "volume", "time_diff", "price_diff", "rsi", "macd", "stoch_rsi", "bb_upperband", "bb_lowerband", "atr"]]
target = df["close"].shift(-1).ffill()  # Use .ffill() to fill NaN values

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Define the model
kernel = C(1.0, (1e-4, 1e2)) * RBF(1, (1e-4, 1e10))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)

# Set up the parameter grid
param_grid = {
    'kernel__k1__constant_value': [0.1, 1, 10, 100],
    'kernel__k2__length_scale': [1, 10, 100, 1000]
}

# Perform grid search
grid_search = GridSearchCV(estimator=gpr, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train_imputed, y_train)

# Print the best parameters found by GridSearchCV
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best MSE: {-grid_search.best_score_}")

# Use the best estimator to make predictions
best_gpr = grid_search.best_estimator_
y_pred, sigma = best_gpr.predict(X_test_imputed, return_std=True)

# Calculate and print metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Predict next candle close price
next_features = features.iloc[-1:].copy()  # Create a DataFrame with the same feature names
next_close_pred, next_sigma = best_gpr.predict(next_features, return_std=True)
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

# Create DataFrame for results
results_df = pd.DataFrame({
    'Real Data': y_test,
    'Predicted Data': y_pred,
    'Difference Percent': ((y_test - y_pred) / y_test) * 100
})

result_file_path = os.path.join(script_dir, 'prediction_result.csv')
results_df.to_csv(result_file_path, index=False)
