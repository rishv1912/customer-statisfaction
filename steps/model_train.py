import logging
import pandas as pd
from zenml import step

import mlflow
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig


from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, config: ModelNameConfig) -> RegressorMixin:
    """
    Trains the model on the ingested data:
    Args:
        X_train: pd.DataFrame,
        X_test : pd.DataFrame,
        y_train : pd.series,
        y_test : pd.series,
    """

    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported")

    except Exception as e:
        logging.error(e)
        raise e
