import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


data = pd.read_csv('goog.us.txt', parse_dates=['Date'], index_col='Date')
data.sort_index(inplace=True)


data['Prev_Close'] = data['Close'].shift(1)
data['MA_5'] = data['Close'].rolling(window=5).mean()
data['MA_20'] = data['Close'].rolling(window=20).mean()
data['Volume_MA_10'] = data['Volume'].rolling(window=10).mean()

# The target is today's close, so we should drop rows with NaNs first
data.dropna(inplace=True)

target = data['Close']
features = data[['Prev_Close', 'MA_5', 'MA_20']]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)


model = RandomForestRegressor(n_estimators=50, max_depth=10,min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
print("Test MSE:", mse)
print("Test R²:", r2)
print("Root Mean Squared Error:", rmse)


# Get the most recent feature row from the dataset
last_features = features.iloc[-1:].copy()
predicted_next_close = model.predict(last_features)[0]
print("Predicted Next Close:", predicted_next_close)
