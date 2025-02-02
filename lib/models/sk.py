from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lib.models.base import BaseClassifier
from lib.path import ModelPath
import joblib


class SKBaseClassifier(BaseClassifier):
    def __init__(self, id, name, model):
        super().__init__(id, name)
        self.model = model

    def fit(self, X, y, *args, **kwargs):
        self.model.fit(X, y)

    def predict_proba(self, X):
        y_pred = self.model.predict_proba(X)
        return y_pred / y_pred.sum(axis=1, keepdims=1)

    def save(self, path: ModelPath):
        joblib.dump(self.model, path.get_path(f"{self.id}.pkl"))

    def load(self, path: ModelPath):
        self.model = joblib.load(path.get_path(f"{self.id}.pkl"))


class LR(SKBaseClassifier):
    def __init__(self, random_state=812):
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(
            solver="saga",
            max_iter=500,
            random_state=random_state,
            C=0.001,
            penalty="l2",
        )

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lr", lr),
            ]
        )
        super().__init__("lr", "LR", pipeline)


class SVM_RBF(SKBaseClassifier):
    def __init__(self, random_state=812):
        from sklearn.svm import SVC

        svm = SVC(
            kernel="rbf",
            max_iter=500,
            probability=True,
            random_state=random_state,
            C=0.1,
            gamma="auto",
        )

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svm", svm),
            ]
        )
        super().__init__("svm-rbf", "SVM-RBF", pipeline)


class SVM_Linear(SKBaseClassifier):
    def __init__(self, random_state=812):
        from sklearn.svm import SVC

        svm = SVC(
            kernel="linear",
            max_iter=500,
            probability=True,
            random_state=random_state,
            C=1000,
        )

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svm", svm),
            ]
        )
        super().__init__("svm-linear", "SVM-Linear", pipeline)


class RF(SKBaseClassifier):
    def __init__(self, random_state=812):
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(
            n_jobs=-1,
            random_state=random_state,
            bootstrap=False,
            criterion="entropy",
            max_features="sqrt",
            n_estimators=200,
        )

        pipeline = Pipeline(
            steps=[
                ("rf", rf),
            ]
        )
        super().__init__("rf", "RF", pipeline)


class LGBM(SKBaseClassifier):
    def __init__(self, random_state=812):
        from lightgbm import LGBMClassifier

        lightgbm = LGBMClassifier(
            n_jobs=-1,
            random_state=random_state,
            boosting_type="gbdt",
            n_estimators=200,
            learning_rate=0.1,
        )

        pipeline = Pipeline(
            steps=[
                ("lightgbm", lightgbm),
            ]
        )
        super().__init__("lgbm", "LGBM", pipeline)


class XGB(SKBaseClassifier):
    def __init__(self, random_state=812):
        from xgboost import XGBClassifier

        xgboost = XGBClassifier(objective="binary:logistic", random_state=random_state)

        pipeline = Pipeline(
            steps=[
                ("xgboost", xgboost),
            ]
        )
        super().__init__("xgb", "XGB", pipeline)


class CatBoost(SKBaseClassifier):
    def __init__(self, random_state=812):
        from catboost import CatBoostClassifier

        catboost = CatBoostClassifier(random_seed=random_state)

        pipeline = Pipeline(
            steps=[
                ("catboost", catboost),
            ]
        )
        super().__init__("catboost", "Catboost", pipeline)
