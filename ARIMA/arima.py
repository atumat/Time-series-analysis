#  Install dependencies
print("Installing dependencies...")
!pip install numpy

# Import libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

print(f"NumPy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")

#  Collect and visualize data
stock = yf.download('AAPL', start='2024-05-19', end='2025-05-19', progress=False)
data = stock['Close']
print("Missing values:", data.isna().sum())

plt.figure(figsize=(10, 6))
plt.plot(data, label='AAPL Adjusted Close')
plt.title('AAPL Stock Prices (May 2024 - May 2025)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

#  Test stationarity
result = adfuller(data)
print('\nADF Test (Original Data):')
print('p-value:', result[1], 'Non-stationary' if result[1] >= 0.05 else 'Stationary')

diff_data = data.diff().dropna()
result_diff = adfuller(diff_data)
print('\nADF Test (Differenced Data):')
print('p-value:', result_diff[1], 'Stationary' if result_diff[1] < 0.05 else 'Non-stationary')

plt.figure(figsize=(10, 6))
plt.plot(diff_data, label='Differenced AAPL Adjusted Close')
plt.title('Differenced AAPL Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price Difference (USD)')
plt.legend()
plt.show()

#  Plot ACF and PACF to select ARIMA parameters
plt.figure(figsize=(12, 4))
plt.subplot(121)
plot_acf(diff_data, lags=20, ax=plt.gca())
plt.title('ACF of Differenced Data')
plt.subplot(122)
plot_pacf(diff_data, lags=20, ax=plt.gca())
plt.title('PACF of Differenced Data')
plt.tight_layout()
plt.show()


order = (1, 1, 1)
print(f"Selected ARIMA order: {order}")

# Fit ARIMA model
arima_model = ARIMA(data, order=order)
model_fit = arima_model.fit()
print(model_fit.summary())

#  Forecast 30 days
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)
forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')

plt.figure(figsize=(10, 6))
plt.plot(data[-60:], label='Actual Prices (Last 60 Days)')
plt.plot(forecast_dates, forecast, label='Forecast', color='red')
plt.title('AAPL Stock Price Forecast (May 20 - June 28, 2025)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

print("\n30-Day Forecast:")
print(pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast}).head())

# Evaluate on test set
train = data[:-30]
test = data[-30:]
train_model = ARIMA(train, order=order)
train_model_fit = train_model.fit()
test_forecast = train_model_fit.forecast(steps=30)
mae = mean_absolute_error(test, test_forecast)
rmse = np.sqrt(mean_squared_error(test, test_forecast))

print('\nEvaluation Metrics (Test Set: Last 30 Days):')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')

plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual Test Data')
plt.plot(test.index, test_forecast, label='Forecast', color='red')
plt.title('Actual vs. Forecasted AAPL Prices (Test Period)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
