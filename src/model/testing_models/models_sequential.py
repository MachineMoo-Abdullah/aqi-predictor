# ============================================================
# TRAIN 3 SEPARATE TIME-SERIES MODELS
#
# Models:
#   1. LSTM
#   2. GRU
#   3. 1D CNN
#
# Input:
#   (samples, 24, 20)
#
# Output:
#   (samples,)
# ============================================================

import joblib
import numpy as np
import tensorflow as tf

from pathlib import Path

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    GRU,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
    .parent
)

MODELS_DIR = BASE_DIR / "models"

X_TRAIN_PATH = MODELS_DIR / "X_train.pkl"
Y_TRAIN_PATH = MODELS_DIR / "y_train.pkl"

X_TEST_PATH = MODELS_DIR / "X_test.pkl"
Y_TEST_PATH = MODELS_DIR / "y_test.pkl"


# Separate folders
LSTM_DIR = MODELS_DIR / "lstm"
GRU_DIR = MODELS_DIR / "gru"
CNN_DIR = MODELS_DIR / "cnn1d"


LSTM_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GRU_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CNN_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

LOOKBACK = 24
N_FEATURES = 20

BATCH_SIZE = 64
EPOCHS = 300

LEARNING_RATE = 1e-3

VALIDATION_SPLIT = 0.20


# ============================================================
# LOAD DATA
# ============================================================

print("\n================================================")
print("LOADING DATA")
print("================================================")


X_train = np.asarray(
    joblib.load(X_TRAIN_PATH),
    dtype=np.float32
)

y_train = np.asarray(
    joblib.load(Y_TRAIN_PATH),
    dtype=np.float32
)

X_test = np.asarray(
    joblib.load(X_TEST_PATH),
    dtype=np.float32
)

y_test = np.asarray(
    joblib.load(Y_TEST_PATH),
    dtype=np.float32
)


print(f"X_train : {X_train.shape}")
print(f"y_train : {y_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_test  : {y_test.shape}")


# ============================================================
# CHECK DATA
# ============================================================

assert X_train.ndim == 3
assert X_test.ndim == 3

assert X_train.shape[1] == LOOKBACK
assert X_train.shape[2] == N_FEATURES

assert X_test.shape[1] == LOOKBACK
assert X_test.shape[2] == N_FEATURES


# ============================================================
# SCALE FEATURES
# ============================================================

print("\n================================================")
print("SCALING")
print("================================================")


scaler = StandardScaler()


# Convert:
# (samples, 24, 20)
#
# to:
# (samples*24, 20)

X_train_2d = X_train.reshape(
    -1,
    N_FEATURES
)

X_test_2d = X_test.reshape(
    -1,
    N_FEATURES
)


# Fit ONLY on training data

X_train_2d = scaler.fit_transform(
    X_train_2d
)

X_test_2d = scaler.transform(
    X_test_2d
)


# Back to 3D

X_train = X_train_2d.reshape(
    -1,
    LOOKBACK,
    N_FEATURES
)

X_test = X_test_2d.reshape(
    -1,
    LOOKBACK,
    N_FEATURES
)


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    LSTM_DIR / "scaler.pkl"
)

joblib.dump(
    scaler,
    GRU_DIR / "scaler.pkl"
)

joblib.dump(
    scaler,
    CNN_DIR / "scaler.pkl"
)


# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks():

    return [

        EarlyStopping(
            monitor="val_loss",
            patience=20,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )

    ]


# ============================================================
# 1. SIMPLE LSTM
# ============================================================

def build_lstm():

    model = Sequential([

        LSTM(
            64,
            input_shape=(
                LOOKBACK,
                N_FEATURES
            )
        ),

        Dropout(0.2),

        Dense(
            32,
            activation="relu"
        ),

        Dense(1)

    ])


    model.compile(

        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="mse",

        metrics=[
            "mae",
            tf.keras.metrics.RootMeanSquaredError(
                name="rmse"
            )
        ]

    )

    return model


# ============================================================
# 2. GRU
# ============================================================

def build_gru():

    model = Sequential([

        GRU(
            64,
            input_shape=(
                LOOKBACK,
                N_FEATURES
            )
        ),

        Dropout(0.2),

        Dense(
            32,
            activation="relu"
        ),

        Dense(1)

    ])


    model.compile(

        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="mse",

        metrics=[
            "mae",
            tf.keras.metrics.RootMeanSquaredError(
                name="rmse"
            )
        ]

    )

    return model


# ============================================================
# 3. 1D CNN
# ============================================================

def build_cnn1d():

    model = Sequential([

        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            input_shape=(
                LOOKBACK,
                N_FEATURES
            )
        ),

        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),

        MaxPooling1D(
            pool_size=2
        ),

        Dropout(0.2),

        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),

        GlobalAveragePooling1D(),

        Dense(
            32,
            activation="relu"
        ),

        Dense(1)

    ])


    model.compile(

        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="mse",

        metrics=[
            "mae",
            tf.keras.metrics.RootMeanSquaredError(
                name="rmse"
            )
        ]

    )

    return model


# ============================================================
# TRAIN LSTM
# ============================================================

print("\n\n")
print("=" * 60)
print("TRAINING LSTM")
print("=" * 60)


lstm = build_lstm()

lstm.summary()


lstm_history = lstm.fit(

    X_train,
    y_train,

    validation_split=VALIDATION_SPLIT,

    shuffle=False,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=get_callbacks(),

    verbose=1

)


# ------------------------------------------------------------
# LSTM TEST
# ------------------------------------------------------------

lstm_pred = lstm.predict(
    X_test,
    verbose=0
).reshape(-1)


lstm_mae = mean_absolute_error(
    y_test,
    lstm_pred
)

lstm_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        lstm_pred
    )
)

lstm_r2 = r2_score(
    y_test,
    lstm_pred
)


# Save LSTM

lstm.save(
    LSTM_DIR / "lstm_model.keras"
)


# Save predictions

joblib.dump(
    lstm_pred,
    LSTM_DIR / "predictions.pkl"
)


print("\n================================================")
print("LSTM RESULT")
print("================================================")

print(f"MAE  : {lstm_mae:.4f}")
print(f"RMSE : {lstm_rmse:.4f}")
print(f"R²   : {lstm_r2:.4f}")

print(
    f"Epochs: {len(lstm_history.history['loss'])}"
)


# ============================================================
# TRAIN GRU
# ============================================================

print("\n\n")
print("=" * 60)
print("TRAINING GRU")
print("=" * 60)


gru = build_gru()

gru.summary()


gru_history = gru.fit(

    X_train,
    y_train,

    validation_split=VALIDATION_SPLIT,

    shuffle=False,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=get_callbacks(),

    verbose=1

)


# ------------------------------------------------------------
# GRU TEST
# ------------------------------------------------------------

gru_pred = gru.predict(
    X_test,
    verbose=0
).reshape(-1)


gru_mae = mean_absolute_error(
    y_test,
    gru_pred
)

gru_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        gru_pred
    )
)

gru_r2 = r2_score(
    y_test,
    gru_pred
)


# Save GRU

gru.save(
    GRU_DIR / "gru_model.keras"
)


joblib.dump(
    gru_pred,
    GRU_DIR / "predictions.pkl"
)


print("\n================================================")
print("GRU RESULT")
print("================================================")

print(f"MAE  : {gru_mae:.4f}")
print(f"RMSE : {gru_rmse:.4f}")
print(f"R²   : {gru_r2:.4f}")

print(
    f"Epochs: {len(gru_history.history['loss'])}"
)


# ============================================================
# TRAIN 1D CNN
# ============================================================

print("\n\n")
print("=" * 60)
print("TRAINING 1D CNN")
print("=" * 60)


cnn = build_cnn1d()

cnn.summary()


cnn_history = cnn.fit(

    X_train,
    y_train,

    validation_split=VALIDATION_SPLIT,

    shuffle=False,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=get_callbacks(),

    verbose=1

)


# ------------------------------------------------------------
# CNN TEST
# ------------------------------------------------------------

cnn_pred = cnn.predict(
    X_test,
    verbose=0
).reshape(-1)


cnn_mae = mean_absolute_error(
    y_test,
    cnn_pred
)

cnn_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        cnn_pred
    )
)

cnn_r2 = r2_score(
    y_test,
    cnn_pred
)


# Save CNN

cnn.save(
    CNN_DIR / "cnn1d_model.keras"
)


joblib.dump(
    cnn_pred,
    CNN_DIR / "predictions.pkl"
)


print("\n================================================")
print("1D CNN RESULT")
print("================================================")

print(f"MAE  : {cnn_mae:.4f}")
print(f"RMSE : {cnn_rmse:.4f}")
print(f"R²   : {cnn_r2:.4f}")

print(
    f"Epochs: {len(cnn_history.history['loss'])}"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n\n")
print("=" * 60)
print("ALL THREE MODELS TRAINED")
print("=" * 60)

print("\nLSTM:")
print(f"MAE  = {lstm_mae:.4f}")
print(f"RMSE = {lstm_rmse:.4f}")
print(f"R²   = {lstm_r2:.4f}")

print("\nGRU:")
print(f"MAE  = {gru_mae:.4f}")
print(f"RMSE = {gru_rmse:.4f}")
print(f"R²   = {gru_r2:.4f}")

print("\n1D CNN:")
print(f"MAE  = {cnn_mae:.4f}")
print(f"RMSE = {cnn_rmse:.4f}")
print(f"R²   = {cnn_r2:.4f}")

print("\nModels saved separately:")
print(f"LSTM : {LSTM_DIR}")
print(f"GRU  : {GRU_DIR}")
print(f"CNN  : {CNN_DIR}")