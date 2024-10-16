import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from time import sleep

# Ensure eager execution for debugging
tf.config.run_functions_eagerly(True)

# 1. Simulating Data Stream
def data_stream(n=1000):
    seasonal = np.sin(np.linspace(0, 20, n))
    noise = np.random.normal(0, 0.5, n)
    anomalies = np.random.choice([0, 5, -5], size=n, p=[0.95, 0.025, 0.025])
    return seasonal + noise + anomalies

# 2. Preparing and Scaling Data
def prepare_data(data, lookback):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.reshape(-1, 1))

    sequences = []
    for i in range(len(data) - lookback):
        sequences.append(data[i:i + lookback])

    return np.array(sequences), scaler

# 3. Build LSTM Autoencoder Model
def build_model(input_shape):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        LSTM(32, activation='relu', return_sequences=False),
        RepeatVector(input_shape[0]),
        LSTM(32, activation='relu', return_sequences=True),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(1))
    ])
    model.compile(optimizer='adam', loss='mse')  # Compile the model
    return model

# 4. Anomaly Detection Based on Reconstruction Error
def detect_anomalies(model, data, threshold, scaler):
    print(f"Data shape before prediction: {data.shape}")
    
    # Check if the data shape matches the expected input shape for the model
    if len(data.shape) == 2:
        data = data.reshape((data.shape[0], data.shape[1], 1))  # Ensure shape is (batch_size, timesteps, features)
    
    reconstructed_data = model.predict(data)
    
    # Compute the reconstruction error
    reconstruction_errors = np.mean(np.abs(reconstructed_data - data), axis=1)
    
    anomalies = reconstruction_errors > threshold
    anomaly_indices = np.where(anomalies)[0]
    return anomaly_indices, reconstruction_errors

# 5. Visualization of Data Stream and Anomalies
def plot_data(data, anomalies, error_threshold):
    plt.figure(figsize=(15, 5))
    plt.plot(data, label="Data Stream")
    plt.scatter(anomalies, data[anomalies], color='red', label="Anomalies", zorder=3)
    plt.axhline(error_threshold, color='green', linestyle='--', label="Error Threshold")
    plt.legend()
    plt.show()

# 6. Real-time streaming simulation with LSTM Anomaly Detection
def real_time_stream_anomaly_detection(lookback=50, epochs=5, threshold=0.01):
    data = data_stream(n=1000)
    sequence_data, scaler = prepare_data(data, lookback)
    
    # Build and train the LSTM autoencoder
    model = build_model((lookback, 1))
    model.fit(sequence_data, sequence_data, epochs=epochs, batch_size=32, validation_split=0.1, verbose=1)
    
    stream_batch_size = 100
    window = deque(maxlen=lookback)
    
    for i in range(0, len(data), stream_batch_size):
        stream_window = data[i:i + stream_batch_size]
        window.extend(stream_window)
        
        if len(window) >= lookback:
            batch_data = np.array(window).reshape(-1, 1)
            batch_sequences, _ = prepare_data(batch_data, lookback)
            
            # Detect anomalies in this batch
            anomalies, errors = detect_anomalies(model, batch_sequences, threshold, scaler)
            
            # Plot data with anomalies
            plot_data(window, anomalies, threshold)
        
        sleep(0.5)  # Simulating real-time data streaming

# 7. Execute real-time streaming anomaly detection
real_time_stream_anomaly_detection()
