from lib.evaluations.metrics import evaluate_all_metrics
from lib.evaluations.predictions import load_predictions
from lib.models.ensemble import Ensemble
from lib.models.nn import NN
from lib.models.sk import LGBM
from lib.path import MetricsPath, ModelPath, Phase, PredictionPath
from lib.utils import init_libraries


init_libraries()

nn = NN()
lgbm = LGBM()
ensemble = Ensemble([nn, lgbm], [26, 24])
models = [nn, lgbm, ensemble]

for model in models:
    print(f"=================== Evaluating {model.id} ===================")

    y_true, y_pred = load_predictions(PredictionPath(model, Phase.TESTING))

    evaluate_all_metrics(y_true, y_pred, MetricsPath(model, Phase.TESTING))
