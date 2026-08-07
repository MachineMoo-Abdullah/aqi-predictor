from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)


def train(model,
          X_train,
          y_train,
          validation_split=0.2):

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

    history = model.fit(

        X_train,

        y_train,

        validation_split=validation_split,

        epochs=300,

        batch_size=64,

        callbacks=callbacks,

        verbose=1
    )

    return history