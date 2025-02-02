from lib.models.base import BaseClassifier
from lib.path import ModelPath
import numpy as np


class Ensemble(BaseClassifier):
    def __init__(self, models: list[BaseClassifier], weights: list[float]):
        super().__init__(
            f"ensemble-{'-'.join([model.id for model in models])}", "Ensemble"
        )
        self.models = models
        self.weights = weights

    def fit(self, X, y, *args, **kwargs):
        for model in self.models:
            model.fit(X, y, *args, **kwargs)

    def predict_proba(self, X):
        y_pred = np.average(
            [model.predict_proba(X)[:, 1] for model in self.models],
            axis=0,
            weights=self.weights,
        ).reshape((-1, 1))
        return np.hstack((1 - y_pred, y_pred))

    def save(self, path: ModelPath):
        for model in self.models:
            model.save(path)

    def load(self, path: ModelPath):
        for model in self.models:
            model.load(path)
