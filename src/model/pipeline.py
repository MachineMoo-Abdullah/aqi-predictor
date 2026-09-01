from .dl_train_model import train_model
from .incremental_training import incremental_train
from .evaluate_model import evaluate_model


def initial_pipeline():

    train_model()

    evaluate_model()


def incremental_pipeline():

    incremental_train()

    evaluate_model()


if __name__ == "__main__":

    incremental_pipeline()