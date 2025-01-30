import numpy as np
from numpy.linalg import inv
import coordinateconverter as converter
import time

# Initial error covariance matrix  
variance = 3000
P = np.power(variance, 2) * np.identity(4)

# Errors during estimation
q = 10
Q = q * np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

def filter(X, z, error, deltaT):
    global P
    angle = z[3][0]

    # State matrix (F)
    F = np.array([[1, 0, 0, deltaT * np.cos(angle)],
                  [0, 1, 0, deltaT * np.sin(angle)],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # Errors during measurement from the GPS library (R)
    R = np.array([[np.power(error.get("x"), 2), 0, 0, 0],
                  [0, np.power(error.get("y"), 2), 0, 0],
                  [0, 0, np.power(5, 2), 0],
                  [0, 0, 0, 0]])

    # Prediction step
    X = F.dot(X)
    P = F.dot(P).dot(F.T) + Q
    pre_x, pre_y = X[0][0], X[1][0]

    # Kalman Gain (K)
    H = np.identity(4)
    S = (H.T).dot(P).dot(H.T) + R
    K = P.dot(H.T).dot(inv(S))

    # Measurement's vector (Y)
    Y = H.dot(z)

    # Update the predicted state using Kalman Gain
    X = X + K.dot(Y - H.dot(X))

    # Update the Process Covariance Matrix
    P = (np.identity(len(K)) - K.dot(H)).dot(P)

    # Convert UTM back to latitude and longitude
    lat, lon = converter.toLatLon(X[0][0], X[1][0])

    # Prediction lat, lon
    pre_lat, pre_lon = converter.toLatLon(pre_x, pre_y)

    return lat, lon, pre_lat, pre_lon, X
