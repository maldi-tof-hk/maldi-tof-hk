from lib.analyses.shap import analyze_all_shap
from lib.data import load_spectra, train_val_split
from lib.models.ensemble import Ensemble
from lib.models.nn import NN
from lib.models.sk import LGBM, LR, RF, SVM_RBF, XGB, CatBoost, SVM_Linear
from lib.path import ModelAnalysisPath, ModelPath, Phase
from lib.utils import init_libraries, random_choice


init_libraries()

df = load_spectra()

X_train, y_train, X_val, y_val = train_val_split(df)

background_data = random_choice(X_train, 1500)
subset = random_choice(X_val, 1500)
nn = NN()
lgbm = LGBM()
ensemble = Ensemble([nn, lgbm], [26, 24])
models = [
    nn,
    lgbm,
    ensemble,
    XGB(),
    CatBoost(),
    LR(),
    RF(),
    SVM_Linear(),
    SVM_RBF(),
]

for model in models:
    print(f"=================== Analyzing {model.id} ===================")

    if model != ensemble:
        # Loading not required for ensemble
        model.load(ModelPath(Phase.TRAINING))

    shap_values = model.compute_shap(subset, background_data)

    analyze_all_shap(shap_values, subset, ModelAnalysisPath(model, Phase.VALIDATION))
