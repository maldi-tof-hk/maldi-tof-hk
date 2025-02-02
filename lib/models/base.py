from sklearn.base import BaseEstimator, ClassifierMixin
import abc

from lib.path import ModelPath


class BaseClassifier(BaseEstimator, ClassifierMixin, metaclass=abc.ABCMeta):
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name

    def predict(self, X):
        return self.predict_proba(X)[:, 1] > 0.5

    @abc.abstractmethod
    def predict_proba(self, X):
        raise NotImplementedError()

    @abc.abstractmethod
    def fit(self, X, y, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, path: ModelPath):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, path: ModelPath):
        raise NotImplementedError()
