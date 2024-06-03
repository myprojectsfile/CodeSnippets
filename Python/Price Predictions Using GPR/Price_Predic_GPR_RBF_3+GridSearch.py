import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

script_dir = os.path.dirname(os.path.realpath(__file__))
read_data_file_path = os.path.join(script_dir, 'BTC-5m-500-OHLCV.csv')

data = pd.read_csv(read_data_file_path)

# Create DataFrame
df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume"])
df["open_time"] = pd.to_datetime(df["open_time"])

# Feature engineering
df['sma_5'] = df['close'].rolling(window=5).mean()
df['sma_10'] = df['close'].rolling(window=10).mean()
df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

# Drop NaN values resulting from rolling calculations
df.dropna(inplace=True)

# Features and target
features = df[["open", "high", "low", "close", "volume", "sma_5", "sma_10", "vwap"]]
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

# Create pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('gpr', gpr)
])

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
logger.info(f'Cross-Validation RMSE: {cv_rmse.mean()}')

# Hyperparameter tuning
param_grid = {
    'gpr__alpha': [1e-2, 1e-3, 1e-4],
    'gpr__kernel': [
        C(1.0, (1e-4, 1e2)) * RBF(length_scale, (1e-4, 1e10)) for length_scale in [1, 10, 100]
    ]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_imputed, y_train)
best_model = grid_search.best_estimator_
logger.info(f'Best model: {best_model}')

# Train the best model
best_model.fit(X_train_imputed, y_train)

# Save the model
model_path = os.path.join(script_dir, 'best_gpr_model.joblib')
joblib.dump(best_model, model_path)

# Predictions
y_pred, sigma = best_model.predict(X_test_imputed, return_std=True)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
logger.info(f"Mean Squared Error: {mse}")
logger.info(f"Root Mean Squared Error: {rmse}")

# Predict next candle close price
next_features = features.iloc[-1:].copy()  # Create a DataFrame with the same feature names
next_close_pred, next_sigma = best_model.predict(next_features, return_std=True)
logger.info(f"Next Candle Close Price Prediction: {next_close_pred[0]} ± {next_sigma[0]}")

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
logger.info(f"Baseline Model MSE: {baseline_mse}")
logger.info(f"Baseline Model RMSE: {baseline_rmse}")

# Create DataFrame for results
results_df = pd.DataFrame({
    'Real Data': y_test,
    'Predicted Data': y_pred,
    'Difference' : y_test - y_pred,
    'Difference Percent': ((y_test - y_pred) / y_test) * 100
})

result_file_path = os.path.join(script_dir, 'prediction_result.csv')
results_df.to_csv(result_file_path, index=False)
