import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


data = pd.read_csv('Stock_Market_Data.csv', parse_dates=['Date'], index_col='Date')
data.sort_index(inplace=True)


data['Prev_Close'] = data['Close'].shift(1)
data['MA_5'] = data['Close'].rolling(window=5).mean()
data['MA_20'] = data['Close'].rolling(window=20).mean()
#data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
#data['Volume_MA_10'] = data['Volume'].rolling(window=10).mean()

data.dropna(inplace=True)

features = data[['Close', 'Prev_Close', 'MA_5', 'MA_20']].values
target = data['Close'].values

# Scale features for LSTM 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

sequence_length = 600  # Number of days to look back to predict the next day's close
X = []
y = []

for i in range(sequence_length, len(scaled_features)):
    # Use previous 'sequence_length' days as input features
    X.append(scaled_features[i-sequence_length:i])
    # The target is the current day closing price
    y.append(scaled_features[i, 0])  # 0 is the index of 'Close' in features

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(Dropout(0.2))
model.add(LSTM(units=100))
#model.add(Dropout(0.2))
model.add(Dense(1))  # Predict a single value (next day's close)

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, shuffle=False)

y_pred = model.predict(X_test)

dummy = np.zeros((len(y_pred), scaled_features.shape[1]))
dummy[:, 0] = y_pred[:, 0]
y_pred_original = scaler.inverse_transform(dummy)[:, 0]

dummy_test = np.zeros((len(y_test), scaled_features.shape[1]))
dummy_test[:, 0] = y_test
y_test_original = scaler.inverse_transform(dummy_test)[:, 0]

mse = mean_squared_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)
rmse = np.sqrt(mse)

print("Test MSE:", mse)
print("Test R²:", r2)
print("Root Mean Squared Error:", rmse)

last_sequence = scaled_features[-sequence_length:]  
last_sequence = np.expand_dims(last_sequence, axis=0)  # reshape for model input
next_day_pred_scaled = model.predict(last_sequence)

# Inverse-transform the prediction
dummy_future = np.zeros((1, scaled_features.shape[1]))
dummy_future[0,0] = next_day_pred_scaled[0,0]
predicted_next_close = scaler.inverse_transform(dummy_future)[0,0]
print("Predicted Next Close:", predicted_next_close)
