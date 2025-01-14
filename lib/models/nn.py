from lib.models.base import BaseClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, LeakyReLU
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import Precision, Recall, FBetaScore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import numpy as np

from lib.path import ModelPath

model_metrics = {
    "Accuracy": "accuracy",
    "Precision": Precision(name="precision"),
    "Recall": Recall(name="recall"),
    "F-beta": FBetaScore(name="f_beta", beta=1.0, threshold=0.5),
}


def build_model():

    model = Sequential()
    model.add(InputLayer(input_shape=(6000,)))

    model.add(Dense(6000))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(6000))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(6000))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(2000))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=AdamW(
            learning_rate=0.0005,
        ),
        metrics=list(model_metrics.values()),
    )

    return model


class NN(BaseClassifier):

    def __init__(self):
        super().__init__("nn")
        self.model = build_model()

    def predict_proba(self, X):
        y_pred = self.model.predict(X)
        return np.hstack((1 - y_pred, y_pred))

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        path: ModelPath | None = None,
        *args,
        **kwargs
    ):
        if len(y.shape) <= 1:
            y = y.reshape((-1, 1))

        if y_val is not None and len(y_val.shape) <= 1:
            y_val = y_val.reshape((-1, 1))

        if path is None:
            callbacks = [EarlyStopping(patience=50)]
        else:
            callbacks = [
                EarlyStopping(patience=50),
                ModelCheckpoint(
                    path.get_path("nn.keras"),
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1,
                ),
            ]

        if X_val is None or y_val is None:
            return self.model.fit(
                X, y, batch_size=128, epochs=1000, callbacks=callbacks, verbose=2
            )
        else:
            return self.model.fit(
                X,
                y,
                validation_data=(X_val, y_val),
                batch_size=128,
                epochs=1000,
                callbacks=callbacks,
                verbose=2,
            )

    def save(self, path: ModelPath):
        self.model.save(path.get_path("nn.keras"))

    def load(self, path: ModelPath):
        self.model = load_model(path.get_path("nn.keras"))
