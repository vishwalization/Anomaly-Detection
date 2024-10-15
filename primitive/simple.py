import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from ipywidgets import interact

def dataStream(N = 1000):
    '''
        TODO: Generate a stream of data 
        \param N Number of data points to generate

        \return Data stream with seasonal, noise and anomalies added
    '''
    seasonal = np.sin(np.linspace(0, 20, N)) 
    noise = np.random.normal(0, 0.5, N)  # Random noise
    anomalies = np.random.choice([0, 5, -5], size=N, p=[0.95, 0.025, 0.025]) 

    return seasonal + noise + anomalies


def anomalyDetection(stream, window_size=20, threshold=3):
    '''
        TODO: Detect anomalies 
        \param stream Data stream 
        \param window_size Size of the deque 
        \param threshold Number of standard deviations away from mean to get(for a data point) flagged as anomaly

        \return anomalies List of index and value each detected anomaly
    '''
    window = deque(maxlen=window_size)
    anomalies = []

    for i, point in enumerate(stream):
        if len(window) == window_size:
            mean = np.mean(window)
            std_dev = np.std(window)
            if abs(point - mean) > threshold * std_dev:
                anomalies.append((i, point))  
        window.append(point)

    return anomalies


def plotData(step):
        '''
        TODO: Slider for visualization 
        \param step Number of data points to display (dynamically updated)
    '''

    plt.clf()  
    x = np.arange(step)
    y = data[:step]
    anomalies = anomalyDetection(y)

    plt.figure(figsize=(25,5))
    plt.plot(x, y, label='Data Stream')
    # anomalies identified as red dots
    if anomalies:                                           
        anomaly_indices, anomaly_values = zip(*anomalies)  
        plt.scatter(anomaly_indices, anomaly_values, color='red', label='Anomalies') 

    plt.xlim(0, len(data))
    plt.ylim(-8, 8)
    plt.legend()
    plt.show()



# Creating data stream and Visualisation with a slider
data = dataStream(1500)
interact(plotData, step=(1, len(data)))
