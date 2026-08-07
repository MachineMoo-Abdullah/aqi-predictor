import tensorflow as tf
from tensorflow.keras import layers, Model


def build_model(lookback, n_features):

    inputs = layers.Input(
        shape=(lookback, n_features)
    )

    x = layers.LSTM(
        128,
        return_sequences=True
    )(inputs)

    x = layers.Dropout(0.30)(x)

    x = layers.LSTM(
        64,
        return_sequences=True
    )(x)

    x = layers.Dropout(0.20)(x)

    attention = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=32
    )(x, x)

    x = layers.Add()([x, attention])

    x = layers.LayerNormalization()(x)

    x = layers.LSTM(32)(x)

    x = layers.Dense(
        64,
        activation="gelu"
    )(x)

    x = layers.Dropout(0.20)(x)

    outputs = layers.Dense(1)(x)

    model = Model(
        inputs,
        outputs
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),

        loss=tf.keras.losses.Huber(),

        metrics=[

            "mae",

            tf.keras.metrics.RootMeanSquaredError(
                name="rmse"
            )

        ]

    )

    return model