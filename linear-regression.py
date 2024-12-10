import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('goog.us.txt', parse_dates=['Date'], index_col='Date')

# Sort data by date just to be sure
data.sort_index(inplace=True)

# - Previous day's close price
# - A 5-day moving average 
# - A 20-day moving average 
data['Prev_Close'] = data['Close'].shift(1)
data['MA_5'] = data['Close'].rolling(window=5).mean()
data['MA_20'] = data['Close'].rolling(window=20).mean()


# The target variable will be today's close price
target = data['Close']

# Drop rows with NaN values that result from shifting and rolling
data = data.dropna()

# Re-define target and features after dropping NaNs
target = data['Close']
features = data[['Prev_Close', 'MA_5', 'MA_20']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Predict the next day's close price
last_features = features.iloc[-1:].copy()
# last_features corresponds to the most recent known day’s Prev_Close, MA_5, and MA_20.

predicted_next_close = model.predict(last_features)[0]
print("Predicted Next Close:", predicted_next_close)
