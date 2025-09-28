from lib.analyses.pseudogel import analyze_pseudogel_three_pane
from lib.data import df_to_xy, load_spectra
from lib.models.ensemble import Ensemble
from lib.models.nn import NN
from lib.models.sk import LGBM
from lib.path import AnalysisPath, ModelPath, Phase
from lib.utils import init_libraries


init_libraries()

df = load_spectra(path="data/external_processed")

X, y = df_to_xy(df)

nn = NN()
lgbm = LGBM()
ensemble = Ensemble([nn, lgbm], [26, 24])
models = [nn, lgbm, ensemble]

for model in models:
    print(f"=================== Evaluating {model.id} ===================")

    if model != ensemble:
        # Loading not required for ensemble
        model.load(ModelPath(Phase.TRAINING))

    y_pred = model.predict_proba(X)[:, 1]

    centers = [1000, 1172, 1526]
    for center in centers:
        analyze_pseudogel_three_pane(
            X[y == 0],
            X[(y == 1) & (y_pred > 0.5)],
            X[(y == 1) & (y_pred < 0.5)],
            center,
            AnalysisPath(Phase.EXTERNAL),
            sample_n=200,
        )
