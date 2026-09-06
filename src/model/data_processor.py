
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def create_3day_sequences(
    df,
    feature_columns,
    target_column="daily_avg_AQI",
    sequence_length=24
):

    X = []
    y = []

    for i in range(len(df) - sequence_length - 72):

        # Previous 24 hours
        X_sequence = df[
            feature_columns
        ].iloc[
            i:i + sequence_length
        ].values

        # First future day
        target_start = i + sequence_length

        # Day 1, Day 2, Day 3
        target = df[
            target_column
        ].iloc[
            [
                target_start,
                target_start + 24,
                target_start + 48
            ]
        ].values

        X.append(X_sequence)
        y.append(target)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y