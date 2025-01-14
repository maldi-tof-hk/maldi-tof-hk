from lib.data import kfold_split, load_spectra, train_val_split
from lib.evaluations.metrics import evaluate_all_metrics, evaluate_metrics, save_metrics
from lib.evaluations.predictions import load_predictions, save_predictions
from lib.models.ensemble import Ensemble
from lib.models.nn import NN
from lib.models.sk import LGBM, LR, RF, SVM_RBF, XGB, CatBoost, SVM_Linear
from lib.path import MetricsPath, ModelPath, Phase, PredictionPath
from lib.utils import init_libraries
import tensorflow as tf
import numpy as np


init_libraries()

df = load_spectra()

nn = NN()
lgbm = LGBM()
ensemble = Ensemble([nn, lgbm], [26, 24])
# nn, lgbm, XGB(), CatBoost(), SVM_RBF(), SVM_Linear(), LR(), RF(),
models = [ensemble]

for model in models:
    print(f"=================== Evaluating {model.id} ===================")

    fold_metrics = []

    for fold_id, X_train, y_train, X_val, y_val in kfold_split(df):
        print(
            f"=================== Evaluating {model.id} fold {fold_id} ==================="
        )

        if fold_id >= 4:
            model.load(ModelPath(Phase.CV, fold=fold_id))

            y_pred = model.predict_proba(X_val)[:, 1]

            save_predictions(
                y_val, y_pred, PredictionPath(model, Phase.CV, fold=fold_id)
            )
        else:
            y_true, y_pred = load_predictions(
                PredictionPath(model, Phase.CV, fold=fold_id)
            )

        evaluate_all_metrics(y_val, y_pred, MetricsPath(model, Phase.CV, fold=fold_id))

        fold_metrics.append(
            evaluate_metrics(y_val, y_pred, MetricsPath(model, Phase.CV, fold=fold_id))
        )

        tf.keras.backend.clear_session()

    metrics = np.average(fold_metrics, axis=0)
    save_metrics(metrics, MetricsPath(model, Phase.CV, fold=0))
