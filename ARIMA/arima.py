# Step 1: Install and verify dependencies
try:
    import numpy as np
    import pmdarima
    print(f"NumPy version: {np.__version__}")
    print(f"pmdarima version: {pmdarima.__version__}")
except ImportError:
    print("Installing numpy and pmdarima...")
    !pip uninstall -y numpy pmdarima
    !pip install numpy<2.0.0 pmdarima
    import numpy as np
    import pmdarima
    print(f"NumPy version: {np.__version__}")
    print(f"pmdarima version: {pmdarima.__version__}")

# Import libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 2: Collect AAPL data
stock = yf.download('AAPL', start='2024-05-19', end='2025-05-19')
data = stock['Adj Close']
print("Missing values:", data.isna().sum())

# Step 3: Visualize data
plt.figure(figsize=(10, 6))
plt.plot(data, label='AAPL Adjusted Close')
plt.title('AAPL Stock Prices (May 2024 - May 2025)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Step 4: ADF test for stationarity (original data)
result = adfuller(data, autolag='AIC')
print('\nADF Test (Original Data):')
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Stationarity:', 'Stationary' if result[1] < 0.05 else 'Non-stationary')

# Step 5: Difference data and re-test
diff_data = data.diff().dropna()
plt.figure(figsize=(10, 6))
plt.plot(diff_data, label='Differenced AAPL Adjusted Close')
plt.title('Differenced AAPL Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price Difference (USD)')
plt.legend()
plt.show()

result_diff = adfuller(diff_data, autolag='AIC')
print('\nADF Test (Differenced Data):')
print('ADF Statistic:', result_diff[0])
print('p-value:', result_diff[1])
print('Stationarity:', 'Stationary' if result_diff[1] < 0.05 else 'Non-stationary')

# Step 6: Select ARIMA parameters with auto_arima
model = auto_arima(data, start_p=0, start_q=0, max_p=3, max_q=3, d=1,
                   seasonal=False, trace=True, error_action='ignore',
                   suppress_warnings=True, stepwise=True)
print(model.summary())

# Step 7: Fit ARIMA model
arima_model = ARIMA(data, order=model.order)  # Use auto_arima's order, e.g., (1,1,1)
model_fit = arima_model.fit()
print(model_fit.summary())

# Step 8: Check residuals
residuals = model_fit.resid
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals')
plt.title('Residuals of ARIMA Model')
plt.axhline(0, color='red', linestyle='--')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plot_acf(residuals, lags=20)
plt.title('ACF of Residuals')
plt.show()

lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print('\nLjung-Box Test for Residuals:')
print(lb_test)

# Step 9: Forecast 30 days
forecast_steps = 30
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

last_date = data.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='B')

plt.figure(figsize=(10, 6))
plt.plot(data[-60:], label='Actual AAPL Prices (Last 60 Days)')
plt.plot(forecast_dates, forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_dates, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], 
                 color='red', alpha=0.2, label='95% Confidence Interval')
plt.title('AAPL Stock Price Forecast (May 20 - June 28, 2025)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Step 10: Display forecast
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_mean})
print("\n30-Day Forecast:")
print(forecast_df.head())

# Step 11: Evaluate on test set
train = data[:-30]
test = data[-30:]
train_model = ARIMA(train, order=model.order)
train_model_fit = train_model.fit()
test_forecast = train_model_fit.forecast(steps=30)
mae = mean_absolute_error(test, test_forecast)
rmse = np.sqrt(mean_squared_error(test, test_forecast))

print('\nEvaluation Metrics (Test Set: Last 30 Days):')
print('MAE:', mae)
print('RMSE:', rmse)

plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual Test Data')
plt.plot(test.index, test_forecast, label='Forecast', color='red')
plt.title('Actual vs. Forecasted AAPL Prices (Test Period)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
