from lib.data import load_spectra, xy_split
from lib.evaluations.history import evaluate_all_history
from lib.models.nn import NN
from lib.models.sk import LGBM
from lib.path import MetricsPath, ModelPath, Phase
from lib.utils import init_libraries


init_libraries()

df = load_spectra()

nn = NN()
lgbm = LGBM()
models = [nn, lgbm]
year = 2021

train = df[df['year'] == year]
val = df[df['year'] != year]
print("Training set:", len(train))
print("Validation set:", len(val))
X_train, y_train, X_val, y_val = xy_split(train, val)

for model in models:
    print(f"=================== Training {model.id} ===================")
    if model == nn:
        # NN auto-saves during training via the checkpoint callback
        history = model.fit(X_train, y_train, X_val, y_val, ModelPath(Phase.BY_YEAR, fold=year))
        evaluate_all_history(history.history, MetricsPath(model, Phase.BY_YEAR, fold=year))
    else:
        model.fit(X_train, y_train)
        model.save(ModelPath(Phase.BY_YEAR, fold=year))
