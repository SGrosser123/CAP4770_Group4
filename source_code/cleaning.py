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
data['target'] = (data['adjusted_close'].shift(-1) > data['adjusted_close']).astype(int)

# Drop NaN Values
data = data.dropna()

# Display the updated data
data.head()

# Calculate Simple Moving Averages (SMA)
data['SMA_20'] = data['adjusted_close'].rolling(window=20).mean()
data['SMA_50'] = data['adjusted_close'].rolling(window=50).mean()


# Function to calculate RSI
def calculate_rsi(data, window):
    delta = data['adjusted_close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Create RSI
data['RSI_14'] = calculate_rsi(data, 14)


# Calculate Ichimoku Cloud components
def ichimoku_cloud(data):
    high_9 = data['high'].rolling(window=9).max()
    low_9 = data['low'].rolling(window=9).min()
    high_26 = data['high'].rolling(window=26).max()
    low_26 = data['low'].rolling(window=26).min()
    high_52 = data['high'].rolling(window=52).max()
    low_52 = data['low'].rolling(window=52).min()

    tenkan_sen = (high_9 + low_9) / 2
    kijun_sen = (high_26 + low_26) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((high_52 + low_52) / 2).shift(26)
    chikou_span = data['adjusted_close'].shift(-26)

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span


data['tenkan_sen'], data['kijun_sen'], data['senkou_span_a'], data['senkou_span_b'], data[
    'chikou_span'] = ichimoku_cloud(data)

# Drop rows with NaN values resulting from rolling calculations
data = data.dropna()
