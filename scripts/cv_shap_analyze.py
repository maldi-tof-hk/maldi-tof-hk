from lib.analyses.shap import analyze_all_shap
from lib.data import kfold_split, load_spectra
from lib.models.ensemble import Ensemble
from lib.models.nn import NN
from lib.models.sk import LGBM
from lib.path import ModelAnalysisPath, ModelPath, Phase
from lib.utils import init_libraries, random_choice
import tensorflow as tf


init_libraries()

df = load_spectra()

nn = NN()
lgbm = LGBM()
ensemble = Ensemble([nn, lgbm], [26, 24])
models = [nn, lgbm, ensemble]

for model in models:
    print(f"=================== Evaluating {model.id} ===================")

    for fold_id, X_train, y_train, X_val, y_val in kfold_split(df):
        print(
            f"=================== Evaluating {model.id} fold {fold_id} ==================="
        )

        background_data = random_choice(X_train, 1500)
        subset = random_choice(X_val, 1500)

        model.load(ModelPath(Phase.CV, fold=fold_id))

        shap_values = model.compute_shap(subset, background_data)

        analyze_all_shap(
            shap_values, subset, ModelAnalysisPath(model, Phase.CV, fold=fold_id)
        )

        tf.keras.backend.clear_session()
