import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# 1. Load and preprocess data
data = pd.read_csv(""C:\Users\mukes\OneDrive\Desktop\intern\#3\silver_price_forecast_2026.csv"")
data = data.fillna(method='ffill').drop_duplicates()

# Ensure Date column is datetime and set index
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# 2. Exploratory visualization
plt.figure(figsize=(10,5))
plt.plot(data.index, data['Price'])
plt.title("Silver Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# 3A. Regression model (if extra features exist)
features = [col for col in data.columns if col not in ['Price']]
if features:
    X = data[features]
    y = data['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    reg_preds = reg_model.predict(X_test)

    print("Regression Results:")
    print("MSE:", mean_squared_error(y_test, reg_preds))
    print("MAE:", mean_absolute_error(y_test, reg_preds))
    print("R²:", r2_score(y_test, reg_preds))

    plt.figure(figsize=(10,5))
    plt.plot(y_test.values, label="Actual")
    plt.plot(reg_preds, label="Predicted")
    plt.legend()
    plt.title("Regression: Actual vs Predicted Silver Price")
    plt.show()

# 3B. Time-series model (ARIMA)
ts_model = ARIMA(data['Price'], order=(5,1,0))
ts_fit = ts_model.fit()
forecast = ts_fit.forecast(steps=10)

print("\nARIMA Forecast (next 10 steps):")
print(forecast)

plt.figure(figsize=(10,5))
plt.plot(data.index, data['Price'], label="Historical")
plt.plot(pd.date_range(data.index[-1], periods=10, freq='D'), forecast, label="Forecast")
plt.legend()
plt.title("ARIMA Forecast of Silver Price")
plt.show()