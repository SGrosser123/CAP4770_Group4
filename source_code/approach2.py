import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.callbacks import TensorBoard

# Set up TensorBoard callback
log_dir = "logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Reshape data for LSTM (samples, time steps, features)
X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model_lstm.add(LSTM(50))
model_lstm.add(Dense(1, activation='sigmoid'))

# Compile the model
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_lstm.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test), callbacks=[tensorboard_callback])

# Make predictions
y_pred_lstm = model_lstm.predict(X_test_lstm)

# Evaluate the model
accuracy_lstm = accuracy_score(y_test, (y_pred_lstm > 0.5).astype(int))
precision_lstm = precision_score(y_test, (y_pred_lstm > 0.5).astype(int))
recall_lstm = recall_score(y_test, (y_pred_lstm > 0.5).astype(int))
f1_lstm = f1_score(y_test, (y_pred_lstm > 0.5).astype(int))
roc_auc_lstm = roc_auc_score(y_test, y_pred_lstm)

print(f'LSTM Accuracy: {accuracy_lstm}')
print(f'LSTM Precision: {precision_lstm}')
print(f'LSTM Recall: {recall_lstm}')
print(f'LSTM F1-Score: {f1_lstm}')
print(f'LSTM ROC-AUC: {roc_auc_lstm}')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow

# Initialize relevant columns, reshape data to fit LSTM model, and generate prediction
relevant_columns = max_date_data.iloc[:, [3, 4, 5]].values
reshaped_data = np.reshape(relevant_columns, (relevant_columns.shape[0], 1, relevant_columns.shape[1]))
lstm_prediction = model_lstm.predict(reshaped_data)

# Branching logic to generate a visual corresponding with prediction
fig, ax = plt.subplots()
if lstm_prediction[0][0] >= 0.5:
    rect = Rectangle((0.2, 0.2), width=0.5, height=0.8, edgecolor='green', facecolor='lightgreen')
    ax.add_patch(rect)
    arrow = Arrow(0.45, 0.2, 0, 0.5, width=0.3, color='green')
    ax.add_patch(arrow)
    ax.axis('off')
    plt.title("The Model Predicts This Stock Will Go Up Tomorrow!")
else:
    rect = Rectangle((0.2, 0.2), width=0.5, height=0.8, edgecolor='red', facecolor='salmon')
    ax.add_patch(rect)
    arrow = Arrow(0.45, 0.7, 0, -0.5, width=0.3, color='red')
    ax.add_patch(arrow)
    ax.axis('off')
    plt.title("The Model Predicts This Stock Will Go Down Tomorrow!")

# Number of days to visualize
days_to_visualize = 30

# Subset the last 30 days of data
subset_data = data[-days_to_visualize:].copy()
subset_dates = subset_data['date']

# Ensure we have the same features as used in training
subset_features = subset_data[['SMA_20', 'SMA_50', 'RSI_14']]
subset_features = scaler.transform(subset_features)

# Reshape data for LSTM prediction
subset_features_lstm = np.reshape(subset_features, (subset_features.shape[0], 1, subset_features.shape[1]))

# Make predictions using the trained model
subset_data['predicted'] = model_lstm.predict(subset_features_lstm).flatten()

# Determine if predictions were accurate
subset_data['accurate'] = ((subset_data['predicted'] > subset_data['adjusted_close']) == (subset_data['adjusted_close'].shift(-1) > subset_data['adjusted_close']))
