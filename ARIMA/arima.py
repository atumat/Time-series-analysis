import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Step 1: Collect data
stock = yf.download('AAPL', start='2024-05-19', end='2025-05-19')
data = stock['Adj Close']
print("Missing values:", data.isna().sum())

# Step 2: Visualize
plt.figure(figsize=(10, 6))
plt.plot(data, label='AAPL Adjusted Close')
plt.title('AAPL Stock Prices (May 2024 - May 2025)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Step 3: ADF test (original data)
result = adfuller(data, autolag='AIC')
print('ADF Test Results (Original Data):')
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Lags Used:', result[2])
print('Number of Observations:', result[3])
print('Critical Values:', result[4])
if result[1] < 0.05:
    print("Reject null hypothesis: Data is stationary")
else:
    print("Fail to reject null hypothesis: Data is non-stationary")

# Step 4: Difference and re-run ADF
diff_data = data.diff().dropna()
plt.figure(figsize=(10, 6))
plt.plot(diff_data, label='Differenced AAPL Adjusted Close')
plt.title('Differenced AAPL Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price Difference (USD)')
plt.legend()
plt.show()

result_diff = adfuller(diff_data, autolag='AIC')
print('\nADF Test Results (Differenced Data):')
print('ADF Statistic:', result_diff[0])
print('p-value:', result_diff[1])
print('Lags Used:', result_diff[2])
print('Number of Observations:', result_diff[3])
print('Critical Values:', result_diff[4])
if result_diff[1] < 0.05:
    print("Reject null hypothesis: Differenced data is stationary")
else:
    print("Fail to reject null hypothesis: Differenced data is non-stationary")

# Step 5: Select ARIMA parameters
model = auto_arima(data, start_p=0, start_q=0, max_p=3, max_q=3, d=1,
                   seasonal=False, trace=True, error_action='ignore',
                   suppress_warnings=True, stepwise=True)
print(model.summary())

# Step 6: Fit ARIMA model
arima_model = ARIMA(data, order=(1, 1, 1))
model_fit = arima_model.fit()
print(model_fit.summary())

# Step 7: Check residuals
residuals = model_fit.resid
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals of ARIMA(1,1,1) Model')
plt.axhline(0, color='red', linestyle='--')
plt.show()

plt.figure(figsize=(10, 6))
plot_acf(residuals, lags=20)
plt.title('ACF of Residuals')
plt.show()

lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print('Ljung-Box Test:')
print(lb_test)

# Step 8: Forecast
forecast_steps = 30
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

last_date = data.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='B')

plt.figure(figsize=(10, 6))
plt.plot(data[-60:], label='Actual AAPL Prices (Last 60 Days)')
plt.plot(forecast_dates, forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_dates, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='red', alpha=0.2, label='95% Confidence Interval')
plt.title('AAPL Stock Price Forecast (30 Days)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_mean})
print(forecast_df)

# Step 9: Evaluate
train = data[:-30]
test = data[-30:]
train_model = ARIMA(train, order=(1, 1, 1))
train_model_fit = train_model.fit()
test_forecast = train_model_fit.forecast(steps=30)
mae = mean_absolute_error(test, test_forecast)
rmse = np.sqrt(mean_squared_error(test, test_forecast))

print('Evaluation Metrics:')
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
