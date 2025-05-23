
# Import libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

print(f"NumPy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")

#  Collect and visualize AAPL data
stock = yf.download('AAPL', start='2024-05-19', end='2025-05-19', progress=False)
data = stock['Close']
print("Missing values:", data.isna().sum())

plt.figure(figsize=(10, 6))
plt.plot(data, label='AAPL Close')
plt.title('AAPL Stock Prices (May 2024 - May 2025)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Test stationarity on returns
returns = data.pct_change().dropna() * 100  # Daily percentage returns
result = adfuller(returns)
print('\nADF Test (Returns):')
print('p-value:', result[1], 'Stationary' if result[1] < 0.05 else 'Non-stationary')

plt.figure(figsize=(10, 6))
plt.plot(returns, label='AAPL Daily Returns (%)')
plt.title('AAPL Daily Returns')
plt.xlabel('Date')
plt.ylabel('Return (%)')
plt.legend()
plt.show()

#  Plot ACF and PACF of squared returns for GARCH parameters
squared_returns = returns**2
plt.figure(figsize=(12, 4))
plt.subplot(121)
plot_acf(squared_returns, lags=20, ax=plt.gca())
plt.title('ACF of Squared Returns')
plt.subplot(122)
plot_pacf(squared_returns, lags=20, ax=plt.gca())
plt.title('PACF of Squared Returns')
plt.tight_layout()
plt.show()

garch_order = (1, 1) 
mean_model = 'AR'  
print(f"Selected GARCH order: {garch_order}, Mean model: {mean_model}")

# Fit GARCH model
garch = arch_model(returns, mean=mean_model, lags=1, vol='GARCH', p=garch_order[0], q=garch_order[1], dist='normal')
model_fit = garch.fit(disp='off')
print(model_fit.summary())

#  Forecast 30-day volatility
forecast_steps = 30
forecast = model_fit.forecast(horizon=forecast_steps, method='simulation', simulations=1000)
forecast_vol = np.sqrt(forecast.variance.values[-1, :])  # Volatility (standard deviation)

forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')

plt.figure(figsize=(10, 6))
plt.plot(returns[-60:], label='Historical Returns (%)')
plt.plot(forecast_dates, forecast_vol, label='Forecasted Volatility (%)', color='red')
plt.title('AAPL Volatility Forecast (May 20 - June 28, 2025)')
plt.xlabel('Date')
plt.ylabel('Volatility (%)')
plt.legend()
plt.show()

print("\n30-Day Volatility Forecast:")
print(pd.DataFrame({'Date': forecast_dates, 'Forecast Volatility (%)': forecast_vol}).head())

#  Evaluate on test set
train_returns = returns[:-30]
test_returns = returns[-30:]
train_model = arch_model(train_returns, mean=mean_model, lags=1, vol='GARCH', p=garch_order[0], q=garch_order[1], dist='normal')
train_model_fit = train_model.fit(disp='off')
test_forecast = train_model_fit.forecast(horizon=30, method='simulation', simulations=1000)
test_forecast_vol = np.sqrt(test_forecast.variance.values[-1, :])

# Evaluate volatility forecast against realized volatility (absolute returns)
realized_vol = np.abs(test_returns)
mae = mean_absolute_error(realized_vol, test_forecast_vol)
rmse = np.sqrt(mean_squared_error(realized_vol, test_forecast_vol))

print('\nEvaluation Metrics (Test Set: Last 30 Days):')
print(f'MAE (Volatility): {mae:.2f}%')
print(f'RMSE (Volatility): {rmse:.2f}%')

plt.figure(figsize=(10, 6))
plt.plot(test_returns.index, realized_vol, label='Realized Volatility (%)')
plt.plot(test_returns.index, test_forecast_vol, label='Forecasted Volatility (%)', color='red')
plt.title('Actual vs. Forecasted AAPL Volatility (Test Period)')
plt.xlabel('Date')
plt.ylabel('Volatility (%)')
plt.legend()
plt.show()
