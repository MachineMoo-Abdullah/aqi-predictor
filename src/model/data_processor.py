import numpy as np
from sklearn.preprocessing import StandardScaler

LOOKBACK = 24

def create_sequences(X, y):

    X_seq = []
    y_seq = []

    for i in range(LOOKBACK, len(X)):
        X_seq.append(X.iloc[i-LOOKBACK:i].values)
        y_seq.append(y.iloc[i])

    return np.array(X_seq), np.array(y_seq)


def scale_train_test(X_train, X_test):

    scaler = StandardScaler()

    train_shape = X_train.shape
    test_shape = X_test.shape

    X_train = scaler.fit_transform(
        X_train.reshape(-1, train_shape[-1])
    ).reshape(train_shape)

    X_test = scaler.transform(
        X_test.reshape(-1, test_shape[-1])
    ).reshape(test_shape)

    return X_train, X_test, scaler


def scale_full_data(X):

    scaler = StandardScaler()

    shape = X.shape

    X = scaler.fit_transform(
        X.reshape(-1, shape[-1])
    ).reshape(shape)

    return X, scaler