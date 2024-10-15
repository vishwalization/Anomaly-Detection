import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from ipywidgets import interact

def dataStream(n=1000):
    '''
        input : number of data points
        return : data stream with amalgamation of random noise and anomalies 
    '''
    seasonal = np.sin(np.linspace(0, 20, n)) 
    noise = np.random.normal(0, 0.5, n)  # Random noise
    anomalies = np.random.choice([0, 5, -5], size=n, p=[0.95, 0.025, 0.025]) 

    return seasonal + noise + anomalies


def anomalyDetection(stream, window_size=20, threshold=3):
    '''
        input : datastream, window_size : for analysing batch of data, threshold : 
        number of standard deviations for a data point to be away from its mean to 
        be flaged as an anomaly 
        return : list of tuple of index and the point flaged as anomaly
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
    plt.clf()  
    x = np.arange(step)
    y = data[:step]
    anomalies = anomalyDetection(y)

    plt.figure(figsize=(25,5))
    plt.plot(x, y, label='Data Stream')
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
