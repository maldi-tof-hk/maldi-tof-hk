from lib.data import load_spectra, train_val_split
from lib.evaluations.history import evaluate_all_history
from lib.models.nn import NN
from lib.models.sk import LGBM, LR, RF, SVM_RBF, XGB, CatBoost, SVM_Linear
from lib.path import MetricsPath, ModelPath, Phase
from lib.utils import init_libraries


init_libraries()

df = load_spectra()

X_train, y_train, X_val, y_val = train_val_split(df)

nn = NN()
models = [nn, LGBM(), XGB(), CatBoost(), SVM_RBF(), SVM_Linear(), LR(), RF()]

for model in models:
    print(f"=================== Training {model.id} ===================")
    if model == nn:
        # NN auto-saves during training via the checkpoint callback
        history = model.fit(X_train, y_train, X_val, y_val, ModelPath(Phase.TRAINING))
        evaluate_all_history(history.history, MetricsPath(model, Phase.TRAINING))
    else:
        model.fit(X_train, y_train)
        model.save(ModelPath(Phase.TRAINING))
