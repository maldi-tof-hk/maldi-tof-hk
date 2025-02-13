from lib.data import load_spectra, train_val_split
from lib.evaluations.history import evaluate_all_history
from lib.evaluations.metrics import evaluate_all_metrics
from lib.evaluations.predictions import save_predictions
from lib.models.nn import MiniNN
from lib.path import MetricsPath, ModelPath, Phase, PredictionPath
from lib.utils import init_libraries

# This script only trains a small neural network with 3 inputs.
# This is used to illustrate the effect of less significant features on model performance.

init_libraries()

df = load_spectra()

X_train, y_train, X_val, y_val = train_val_split(df)

model = MiniNN()

print(f"=================== Training {model.id} ===================")
# NN auto-saves during training via the checkpoint callback
history = model.fit(X_train, y_train, X_val, y_val, ModelPath(Phase.TRAINING))
evaluate_all_history(history.history, MetricsPath(model, Phase.TRAINING))

print(f"=================== Evaluating {model.id} ===================")

model.load(ModelPath(Phase.TRAINING))

y_pred = model.predict_proba(X_val)[:, 1]

save_predictions(y_val, y_pred, PredictionPath(model, Phase.VALIDATION))

evaluate_all_metrics(y_val, y_pred, MetricsPath(model, Phase.VALIDATION))
