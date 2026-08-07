from prepare_data import prepare_dataset
from data_processor import create_sequences, scale_train_test
from model_builder import build_model
from trainer import train
from saver import save_training_model

from sklearn.model_selection import train_test_split


def train_model():

    print("Loading dataset...")

    X, y = prepare_dataset()

    X_seq, y_seq = create_sequences(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq,
        y_seq,
        test_size=0.2,
        shuffle=False
    )

    X_train, X_test, scaler = scale_train_test(
        X_train,
        X_test
    )

    LOOKBACK = 24

    model = build_model(
        LOOKBACK,
        X_train.shape[2]
    )

    train(
        model,
        X_train,
        y_train
    )

    save_training_model(
        model,
        scaler,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("Initial training completed.")