from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prepare_data import prepare_dataset

from sklearn.model_selection import train_test_split

from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score
)

import joblib

X, y = prepare_dataset()

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    shuffle=False
)

import tensorflow as tf

model = tf.keras.models.load_model(
    "Model/models/complex_dl_model.keras"
)
pred = model.predict(X_test)

print("MAE :", mean_absolute_error(y_test, pred))

print("RMSE:", mean_squared_error(y_test, pred)**0.5)

print("R²  :", r2_score(y_test, pred))