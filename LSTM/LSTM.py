import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from pykalman import KalmanFilter

def simulate_data_stream(n_points=1000, seasonality=100, noise=1, anomaly_rate=0.01):
    '''
    TODO: Simulate a data stream with seasonal patterns, noise, and anomalies.
    \param n_points: Number of data points to simulate.
    \param seasonality: Period of the seasonal pattern.
    \param noise: Standard deviation of the noise to be added.
    \param anomaly_rate: Proportion of data points to be treated as anomalies.

    \return: A numpy array representing the simulated data stream with anomalies.
    '''
    time = np.arange(n_points)
    seasonal_pattern = 10 * np.sin(2 * np.pi * time / seasonality)
    noise = np.random.normal(0, noise, n_points)
    data = seasonal_pattern + noise

    anomaly_indices = np.random.choice(np.arange(n_points), int(n_points * anomaly_rate), replace=False)
    anomaly_values = np.random.uniform(20, 30, size=len(anomaly_indices)) 
    data[anomaly_indices] += anomaly_values

    return data

def create_lstm_autoencoder(input_shape):
    '''
    TODO: Create an enhanced LSTM autoencoder model for anomaly detection.
    \param input_shape: Shape of the input data for the LSTM model.

    \return: A compiled LSTM autoencoder model.
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, activation='relu', return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=False),
        tf.keras.layers.RepeatVector(input_shape[0]),
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(128, activation='relu', return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
    ])
    model.compile(optimizer='adam', loss='mse')

    return model

def train_lstm_autoencoder(data_stream, window_size=100):
    '''
        TODO: Train the LSTM autoencoder on the provided data stream.
        \param data_stream: The input time series data for training.
        \param window_size: The size of the sliding window used to create sequences.

        \return: A tuple containing the trained autoencoder and the scaler used for normalization.
    '''
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_stream = scaler.fit_transform(data_stream.reshape(-1, 1))

    sequences = np.array([data_stream[i:i + window_size] for i in range(len(data_stream) - window_size)])
    autoencoder = create_lstm_autoencoder((window_size, 1))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    autoencoder.fit(sequences, sequences,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=[early_stopping])

    return autoencoder, scaler

def detect_anomalies_lstm(autoencoder, data_stream, scaler, window_size=100, alpha=0.05):
    '''
        TODO: Detect anomalies using the trained LSTM autoencoder based on reconstruction error.
        \param autoencoder: The trained LSTM autoencoder model.
        \param data_stream: The input time series data for anomaly detection.
        \param scaler: The scaler used to normalize the input data.
        \param window_size: The size of the sliding window used to create sequences.
        \param alpha: The significance level used to determine anomalies.

        \return: A tuple containing a boolean array indicating detected anomalies and the novelty scores.
    '''
    data_stream_scaled = scaler.transform(data_stream.reshape(-1, 1))

    sequences = np.array([data_stream_scaled[i:i + window_size] for i in range(len(data_stream_scaled) - window_size)])
    reconstructions = autoencoder.predict(sequences)
    mse = np.mean(np.square(sequences - reconstructions), axis=(1, 2))

    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1 (inclusive)")

    novelty_scores = mse / np.mean(mse)
    threshold = np.percentile(novelty_scores, 100 - 100 * alpha)
    anomalies = novelty_scores > threshold

    return anomalies, novelty_scores

def apply_arima_kalman(data_stream, order=(5, 1, 0)):
    '''
        TODO: Applie ARIMA and Kalman Filter for comparison and smoothing of the time series.
        \param data_stream: The input time series data for modeling.
        \param order: The order of the ARIMA model (p,d,q).

        \return: A tuple containing the fitted ARIMA model and the Kalman filter output (state means).
     '''
    model = ARIMA(data_stream, order=order)
    model_fit = model.fit()

    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=data_stream[0],
                      initial_state_covariance=1,
                      observation_covariance=5,
                      transition_covariance=0.01)

    state_means, _ = kf.filter(data_stream)
    
    return model_fit, state_means

def visualize_data_stream(data_stream, anomalies=None, kalman_output=None):
    '''
        Visualize the original data stream along with detected anomalies and Kalman Filter output.
        \param data_stream: The original time series data to be plotted.
        \param anomalies: A boolean array indicating which points are anomalies (optional).
        \param kalman_output: The output from the Kalman filter (optional).

        Displays a plot with sliders for interactive exploration of the time series and detected anomalies.
   '''
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(data_stream, label='Data Stream', color='blue')

    if anomalies is not None:
        anomaly_indices = np.where(anomalies)[0]  # Get indices of anomalies
        ax.scatter(anomaly_indices, 
                   data_stream[anomaly_indices], 
                   color='red', label='Anomalies', zorder=10)  # Plot anomalies directly

    if kalman_output is not None:
        ax.plot(kalman_output, label='Kalman Filter Output', color='green')
    
    ax.set_title('Data Stream with Anomalies and Kalman Filter Output')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.set_xlim(0, len(data_stream))
    ax.set_ylim(np.min(data_stream) - 10, np.max(data_stream) + 10)

    slider_ax = plt.axes([0.15, 0.02, 0.7, 0.03])
    slider = plt.Slider(slider_ax, 'Time', valmin=0,
                        valmax=len(data_stream)-1,
                        valinit=0,
                        orientation='horizontal')
    
    def update(val):
        index = int(slider.val)
        line.set_data(np.arange(index+1), data_stream[:index+1])

        if anomalies is not None:
            anomaly_indices = np.where(anomalies[:index+1])[0]
            anomaly_scatter.set_offsets(np.c_[anomaly_indices, data_stream[anomaly_indices]])

        if kalman_output is not None:
            kalman_line.set_data(np.arange(index+1), kalman_output[:index+1])

        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()



data_stream = simulate_data_stream(n_points=1000, anomaly_rate=0.02)
autoencoder_model, scaler_model = train_lstm_autoencoder(data_stream)
anomalies_lstm, novelty_scores_lstm = detect_anomalies_lstm(autoencoder_model,
                                                              data_stream,
                                                              scaler_model,
                                                              window_size=100,
                                                              alpha=0.05)

model_fit_kalman, kalman_output = apply_arima_kalman(data_stream)
visualize_data_stream(data_stream, anomalies=anomalies_lstm, kalman_output=kalman_output)