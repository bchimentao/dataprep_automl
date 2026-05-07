import pandas as pd
from autogluon.tabular import TabularPredictor

from .base import AutoMLRunner


class AutoGluonRunner(AutoMLRunner):
    def __init__(self, label, output_path, eval_metric="f1", time_limit=300):
        self.label = label
        self.output_path = output_path
        self.eval_metric = eval_metric
        self.time_limit = time_limit
        self.predictor = None

    def train(self, X_train, y_train):
        # Create a training dataframe with features and target.
        train_data = X_train.copy()
        train_data[self.label] = y_train

        self.predictor = TabularPredictor(
            label=self.label,
            path=self.output_path,
            eval_metric=self.eval_metric
        ).fit(
            train_data=train_data,
            time_limit=self.time_limit
        )

    def predict(self, X_test):
        # Return class predictions.
        return self.predictor.predict(X_test)

    def predict_proba(self, X_test):
        # Return class probabilities.
        return self.predictor.predict_proba(X_test)