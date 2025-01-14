from lib.path import PredictionPath
import pandas as pd


def save_predictions(y_true, y_pred, path: PredictionPath):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df.to_csv(path.get_path("predictions.csv"), index=False)


def load_predictions(path: PredictionPath):
    df = pd.read_csv(path.get_path("predictions.csv"))
    return df["y_true"].values, df["y_pred"].values
