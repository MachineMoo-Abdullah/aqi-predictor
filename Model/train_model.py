import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from prepare_data import prepare_dataset

# ----------------------------
# Load Data
# ----------------------------
X, y = prepare_dataset()

# Scale Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

input_dim = X_train.shape[1]

# ----------------------------
# Residual Block
# ----------------------------
def residual_block(x, units, dropout_rate):

    shortcut = x

    x = layers.Dense(units)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(units)(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units)(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("gelu")(x)

    return x

# ----------------------------
# Model
# ----------------------------
inputs = layers.Input(shape=(input_dim,))

x = layers.Dense(512)(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation("gelu")(x)
x = layers.Dropout(0.30)(x)

x = residual_block(x, 512, 0.30)
x = residual_block(x, 256, 0.25)
x = residual_block(x, 128, 0.20)
x = residual_block(x, 64, 0.15)

x = layers.Dense(32, activation="gelu")(x)

outputs = layers.Dense(1, activation="linear")(x)

model = Model(inputs, outputs)

# ----------------------------
# Compile
# ----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.Huber(),
    metrics=["mae"]
)

# ----------------------------
# Callbacks
# ----------------------------
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

# ----------------------------
# Train
# ----------------------------
history = model.fit(

    X_train,
    y_train,

    validation_split=0.2,

    epochs=300,

    batch_size=64,

    callbacks=callbacks,

    verbose=1
)

# ----------------------------
# Save
# ----------------------------
model.save("Model/models/complex_dl_model.keras")

print("Model Saved")