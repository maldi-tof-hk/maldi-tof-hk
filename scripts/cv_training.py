from lib.data import kfold_split, load_spectra
from lib.evaluations.history import evaluate_all_history
from lib.models.nn import NN
from lib.models.sk import LGBM, LR, RF, SVM_RBF, XGB, CatBoost, SVM_Linear
from lib.path import MetricsPath, ModelPath, Phase
from lib.utils import init_libraries


init_libraries()

df = load_spectra()

nn = NN()
models = [nn, LGBM(), XGB(), CatBoost(), SVM_RBF(), SVM_Linear(), LR(), RF()]

for model in models:
    print(f"=================== Training {model.id} ===================")

    for fold_id, X_train, y_train, X_val, y_val in kfold_split(df):
        print(
            f"=================== Training {model.id} fold {fold_id} ==================="
        )

        if model == nn:
            # NN auto-saves during training via the checkpoint callback
            history = model.fit(
                X_train, y_train, X_val, y_val, ModelPath(Phase.CV, fold=fold_id)
            )
            evaluate_all_history(
                history.history, MetricsPath(model, Phase.CV, fold=fold_id)
            )
        else:
            model.fit(X_train, y_train)
            model.save(ModelPath(Phase.CV, fold=fold_id))
