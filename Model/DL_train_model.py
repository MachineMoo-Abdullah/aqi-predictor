import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import numpy as np
import joblib

from prepare_data import prepare_dataset

# ====================================================
# Load Data
# ====================================================

X, y = prepare_dataset()

# ====================================================
# Create sequences (before scaling)
# ====================================================

LOOKBACK = 24

X_seq = []
y_seq = []

for i in range(LOOKBACK, len(X)):
    X_seq.append(X[i - LOOKBACK:i])
    y_seq.append(y[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print(X_seq.shape)
print(y_seq.shape)

# ====================================================
# Train/Test Split
# ====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_seq,
    y_seq,
    test_size=0.2,
    shuffle=False
)

# ====================================================
# Scale (fit ONLY on training data)
# ====================================================

X_train_shape = X_train.shape
X_test_shape = X_test.shape

X_train_flat = X_train.reshape(-1, X_train.shape[-1])
X_test_flat = X_test.reshape(-1, X_test.shape[-1])

scaler = StandardScaler()

X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)

X_train = X_train_flat.reshape(X_train_shape)
X_test = X_test_flat.reshape(X_test_shape)

# ====================================================
# Model
# ====================================================

inputs = layers.Input(
    shape=(LOOKBACK, X_train.shape[2])
)

# LSTM 1
x = layers.LSTM(
    128,
    return_sequences=True
)(inputs)

x = layers.Dropout(0.3)(x)

# LSTM 2
x = layers.LSTM(
    64,
    return_sequences=True
)(x)

x = layers.Dropout(0.2)(x)

# ====================================================
# Self Attention
# ====================================================

attention = layers.MultiHeadAttention(
    num_heads=4,
    key_dim=32
)(x, x)

x = layers.Add()([x, attention])

x = layers.LayerNormalization()(x)

# ====================================================
# Last LSTM
# ====================================================

x = layers.LSTM(32)(x)

x = layers.Dense(
    64,
    activation="gelu"
)(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(1)(x)

model = Model(inputs, outputs)

# ====================================================
# Compile
# ====================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(1e-3),

    loss=tf.keras.losses.Huber(),

    metrics=[
        "mae",
        tf.keras.metrics.RootMeanSquaredError(name="rmse")
    ]
)

model.summary()

# ====================================================
# Callbacks
# ====================================================

callbacks = [

    EarlyStopping(

        monitor="val_loss",

        patience=20,

        restore_best_weights=True

    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=8,

        min_lr=1e-6

    )
]

# ====================================================
# Train
# ====================================================

history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    epochs=300,

    batch_size=64,

    callbacks=callbacks,

    verbose=1
)

# ====================================================
# Evaluate
# ====================================================

loss, mae, rmse = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"Test MAE  : {mae:.4f}")
print(f"Test RMSE : {rmse:.4f}")




pred = model.predict(X_test, verbose=0).flatten()
# Metrics
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ====================================================
# Save
# ====================================================

# ====================================================
# Save Model & Preprocessing
# ====================================================

# Save model
model.save("Model/models/lstm_attention(std).keras")

# Save scaler
joblib.dump(
    scaler,
    "Model/models/scaler.pkl"
)

# Save train/test datasets
joblib.dump(
    X_train,
    "Model/models/X_train.pkl"
)

joblib.dump(
    y_train,
    "Model/models/y_train.pkl"
)

joblib.dump(
    X_test,
    "Model/models/X_test.pkl"
)

joblib.dump(
    y_test,
    "Model/models/y_test.pkl"
)

print("\n===================================")
print("Model Saved Successfully!")
print("Scaler Saved.")
print("Train/Test datasets Saved.")
print("===================================")
