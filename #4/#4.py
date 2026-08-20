import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
data = pd.read_csv("silver_price_forecast_2026.csv")

# 2. Data Cleaning
# Handle missing values (forward fill)
data = data.fillna(method='ffill')

# Remove duplicates
data = data.drop_duplicates()

# Standardize column names (remove spaces, lowercase)
data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

# Quick summary report
print("=== Data Summary ===")
print(data.describe())
print("\nMissing Values:\n", data.isnull().sum())

# 3. Automated Reporting
# Save cleaned dataset
data.to_csv("cleaned_silver_price.csv", index=False)

# Generate summary statistics report
summary_report = data.describe()
summary_report.to_csv("silver_price_summary_report.csv")

# 4. Visual Reporting
plt.figure(figsize=(10,5))
plt.plot(pd.to_datetime(data['date']), data['price'], color='silver')
plt.title("Silver Price Trend (Forecast 2026)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.savefig("silver_price_trend.png")
plt.show()

# 5. Automated Insights
print("\n=== Automated Insights ===")
print(f"Dataset contains {len(data)} records after cleaning.")
print(f"Average Silver Price: {data['price'].mean():.2f}")
print(f"Max Silver Price: {data['price'].max():.2f}")
print(f"Min Silver Price: {data['price'].min():.2f}")