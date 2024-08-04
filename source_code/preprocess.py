from alpha_vantage.timeseries import TimeSeries

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
api_key = 'YKRVEKRBE0PWGPCV'
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')

# Include the close prices and volume
data = data[['4. close', '5. volume']]

# Reverse the order to have oldest to newest
data = data[::-1]

# Reset the index to make 'date' a column
data = data.reset_index()
data.columns = ['date', 'close', 'volume']

# Display the first few rows of the preprocessed dataset
data.head()

# Display the range of dates in the dataset
print(f"Data covers from {data['date'].min()} to {data['date'].max()}")


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


from alpha_vantage.timeseries import TimeSeries

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage premium API key
api_key = 'YKRVEKRBE0PWGPCV'
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily_adjusted(symbol='AAPL', outputsize='full')

# Include the adjusted close prices and volume
data = data[['5. adjusted close', '6. volume']]

# Reverse the order to have oldest to newest
data = data[::-1]

# Reset the index to make 'date' a column
data = data.reset_index()
data.columns = ['date', 'adjusted_close', 'volume']

# Display the first few rows of the preprocessed dataset
data.head()

# Display the range of dates in the dataset
print(f"Data covers from {data['date'].min()} to {data['date'].max()}")


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


from alpha_vantage.timeseries import TimeSeries

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage premium API key
api_key = 'YKRVEKRBE0PWGPCV'
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily_adjusted(symbol='AAPL', outputsize='full')

# Include the adjusted close prices, high, and low prices
data = data[['5. adjusted close', '2. high', '3. low', '6. volume']]
data.columns = ['adjusted_close', 'high', 'low', 'volume']

# Reverse the order to have oldest to newest
data = data[::-1]

# Reset the index to make 'date' a column
data = data.reset_index()
data.columns = ['date', 'adjusted_close', 'high', 'low', 'volume']

# Display the first few rows of the preprocessed dataset
data.head()

# Display the range of dates in the dataset
print(f"Data covers from {data['date'].min()} to {data['date'].max()}")


