from lib.data import load_spectra, train_val_split
from lib.evaluations.metrics import evaluate_all_metrics
from lib.evaluations.predictions import save_predictions
from lib.models.ensemble import Ensemble
from lib.models.nn import NN
from lib.models.sk import LGBM, LR, RF, SVM_RBF, XGB, CatBoost, SVM_Linear
from lib.path import MetricsPath, ModelPath, Phase, PredictionPath
from lib.utils import init_libraries


init_libraries()

df = load_spectra()

X_train, y_train, X_val, y_val = train_val_split(df)

nn = NN()
lgbm = LGBM()
ensemble = Ensemble([nn, lgbm], [26, 24])
models = [nn, lgbm, XGB(), CatBoost(), SVM_RBF(), SVM_Linear(), LR(), RF(), ensemble]

for model in models:
    print(f"=================== Evaluating {model.id} ===================")

    if model != ensemble:
        # Loading not required for ensemble
        model.load(ModelPath(Phase.TRAINING))

    y_pred = model.predict_proba(X_val)[:, 1]

    save_predictions(y_val, y_pred, PredictionPath(model, Phase.VALIDATION))

    evaluate_all_metrics(y_val, y_pred, MetricsPath(model, Phase.VALIDATION))
