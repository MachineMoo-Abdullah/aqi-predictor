import tensorflow as tf
import joblib
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load model
model = tf.keras.models.load_model(
    "Model/models/lstm_attention.keras"
)

# Load exact test set used during training
X_test = joblib.load("Model/models/X_test.pkl")
y_test = joblib.load("Model/models/y_test.pkl")

# Predict
pred = model.predict(X_test, verbose=0).flatten()

# Metrics
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")