from prepare_data import prepare_dataset
from data_processor import create_sequences, scale_full_data
from model_builder import build_model
from trainer import train
from saver import save_candidate


def incremental_train():

    print("Loading complete dataset...")

    X, y = prepare_dataset()

    X_seq, y_seq = create_sequences(
        X,
        y
    )

    X_seq, scaler = scale_full_data(
        X_seq
    )

    LOOKBACK = 24

    model = build_model(
        LOOKBACK,
        X_seq.shape[2]
    )

    train(
        model,
        X_seq,
        y_seq
    )

    save_candidate(
        model,
        scaler
    )

    print("Incremental model trained.")