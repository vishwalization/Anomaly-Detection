import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from time import sleep

tf.config.run_functions_eagerly(True)

def data_stream(n=1000):
    seasonal = np.sin(np.linspace(0, 20, n))
    noise = np.random.normal(0, 0.5, n)
    anomalies = np.random.choice([0, 5, -5], size=n, p=[0.95, 0.025, 0.025])
    return seasonal + noise + anomalies

def prepare_data(data, lookback):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.reshape(-1, 1))

    sequences = []
    for i in range(len(data) - lookback):
        sequences.append(data[i:i + lookback])

    return np.array(sequences), scaler

    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        LSTM(32, activation='relu', return_sequences=False),
        RepeatVector(input_shape[0]),
        LSTM(32, activation='relu', return_sequences=True),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(1))
    ])
    model.compile(optimizer='adam', loss='mse')  
    return model

def detect_anomalies(model, data, threshold, scaler):
    print(f"Data shape before prediction: {data.shape}")
    
    if len(data.shape) == 2:
        data = data.reshape((data.shape[0], data.shape[1], 1))  
    
    reconstructed_data = model.predict(data)
    
    reconstruction_errors = np.mean(np.abs(reconstructed_data - data), axis=1)
    
    anomalies = reconstruction_errors > threshold
    anomaly_indices = np.where(anomalies)[0]
    return anomaly_indices, reconstruction_errors

def plot_data(data, anomalies, error_threshold):
    plt.figure(figsize=(15, 5))
    plt.plot(data, label="Data Stream")
    plt.scatter(anomalies, data[anomalies], color='red', label="Anomalies", zorder=3)
    plt.axhline(error_threshold, color='green', linestyle='--', label="Error Threshold")
    plt.legend()
    plt.show()

    data = data_stream(n=1000)
    sequence_data, scaler = prepare_data(data, lookback)
    
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
            
            anomalies, errors = detect_anomalies(model, batch_sequences, threshold, scaler)
            plot_data(window, anomalies, threshold)
        
        sleep(0.5) 

real_time_stream_anomaly_detection()
