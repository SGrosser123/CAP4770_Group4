import pandas as pd
import numpy as np

# Create Moving Averages
data['SMA_20'] = data['close'].rolling(window=20).mean()
data['SMA_50'] = data['close'].rolling(window=50).mean()

# Function to calculate RSI
def calculate_rsi(data, window):
    delta = data['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Create RSI
data['RSI_14'] = calculate_rsi(data, 14)

# Create Target Variable
data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

# Drop NaN Values
data = data.dropna()

# Display the updated data
data.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Select features and target variable
features = data[['SMA_20', 'SMA_50', 'RSI_14']]
target = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt

# Split the dates for training and testing
train_dates = data['date'][:len(X_train)]
test_dates = data['date'][len(X_train):]

# Plot the closing prices
plt.figure(figsize=(14, 7))
plt.plot(train_dates, data['close'][:len(X_train)], label='Training Data')
plt.plot(test_dates, data['close'][len(X_train):], label='Testing Data')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('AAPL Stock Price - Training vs Testing Data')
plt.legend()
plt.show()
