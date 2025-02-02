from lib.analyses.pseudogel import analyze_pseudogel
from lib.analyses.scatter import aggregate_intensities, analyze_scatter
from lib.data import load_spectra, train_val_split
from lib.path import AnalysisPath, Phase
from lib.utils import init_libraries


init_libraries()

df = load_spectra()

X_train, y_train, X_val, y_val = train_val_split(df)

peak999 = aggregate_intensities(X_val, 999, 1002)
peak1526 = aggregate_intensities(X_val, 1526, 1528)
peak1172 = aggregate_intensities(X_val, 1172, 1175)

analyze_scatter(peak999, peak1526, y_val, AnalysisPath(Phase.VALIDATION))
analyze_scatter(peak999, peak1172, y_val, AnalysisPath(Phase.VALIDATION))
analyze_scatter(peak1526, peak1172, y_val, AnalysisPath(Phase.VALIDATION))

centers = [1000, 1172, 1526]
for center in centers:
    analyze_pseudogel(
        X_val[y_val == 0], X_val[y_val == 1], center, AnalysisPath(Phase.VALIDATION)
    )
