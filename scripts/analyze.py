import numpy as np
from lib.analyses.pseudogel import analyze_pseudogel
from lib.analyses.scatter import analyze_scatter
from lib.analyses.shap import analyze_all_shap
from lib.data import aggregate_intensities, load_spectra, train_val_split
from lib.models.ensemble import Ensemble
from lib.models.nn import NN
from lib.models.sk import LGBM, LR, RF, SVM_RBF, XGB, CatBoost, SVM_Linear
from lib.path import AnalysisPath, ModelAnalysisPath, ModelPath, Phase
from lib.utils import init_libraries, random_choice


init_libraries()

df = load_spectra()

X_train, y_train, X_val, y_val = train_val_split(df)

peak999 = aggregate_intensities(X_val, 999, 1002)
peak999_name = "Feature 999-1002"
peak1526 = aggregate_intensities(X_val, 1526, 1528)
peak1526_name = "Feature 1526-1528"
peak1172 = aggregate_intensities(X_val, 1172, 1175)
peak1172_name = "Feature 1172-1175"

analyze_scatter(
    peak999,
    peak1526,
    y_val,
    x_name=peak999_name,
    y_name=peak1526_name,
    path=AnalysisPath(Phase.VALIDATION),
)
analyze_scatter(
    peak999,
    peak1172,
    y_val,
    x_name=peak999_name,
    y_name=peak1172_name,
    path=AnalysisPath(Phase.VALIDATION),
)
analyze_scatter(
    peak1526,
    peak1172,
    y_val,
    x_name=peak1526_name,
    y_name=peak1172_name,
    path=AnalysisPath(Phase.VALIDATION),
)

centers = [1000, 1172, 1526]
for center in centers:
    analyze_pseudogel(
        X_val[y_val == 0], X_val[y_val == 1], center, AnalysisPath(Phase.VALIDATION)
    )

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
