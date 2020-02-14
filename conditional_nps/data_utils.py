"""
Utility functions for 1d regression experiments
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


def transform_data(X_train, y_train, X_test, y_test):
    """
    Apply feature scaling to the data. Return the standardised and low-dimensional train and
    test sets together with the scaler object for the target values.

    :param X_train: input train data
    :param y_train: train labels
    :param X_test: input test data
    :param y_test: test labels
    :return: X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
    """
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, x_scaler, y_scaler

def x_generator(min_x, max_x, n_points):
    x1 = min_x + 1
    x2 = max_x - 1
    x = (x2 - x1) * np.random.random(n_points - 2) + x1
    x = np.insert(x, 0, values = min_x)
    x = np.append(x, values = max_x)
    x = np.reshape(x, (x.shape[0], 1))
    return x

def noisy_function(x, std):
    y = x**3 + np.random.normal(loc=0, scale=std, size=(x.shape[0],1))
    return y
