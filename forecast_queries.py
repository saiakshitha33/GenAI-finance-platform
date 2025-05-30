import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load historical query log
df = pd.read_csv("query_logs.csv")  # Must have 'ds' (date), 'y' (query volume) columns
# Example rows:
# ds,y
# 2024-01-01,120
# 2024-01-02,135
# ...

# Fit the model
model = Prophet()
model.fit(df)

# Forecast the next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title("Query Volume Forecast")
plt.xlabel("Date")
plt.ylabel("Queries")
plt.show()
