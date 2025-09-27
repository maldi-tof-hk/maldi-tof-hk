from enum import StrEnum
import abc
import typing

if typing.TYPE_CHECKING:
    from lib.models.base import BaseClassifier

output_root = "data"


class Phase(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"
    CV = "cv"
    TESTING = "testing"
    BY_YEAR = "by_year"
    EXTERNAL = "external"


class OutputPath(metaclass=abc.ABCMeta):
    def __init__(self, root: str):
        self.root = root

    @abc.abstractmethod
    def get_subpath(self):
        raise NotImplementedError()

    def get_path(self, file: str):
        return f"{output_root}/{self.root}/{self.get_subpath()}-{file}"


class ModelPath(OutputPath):
    def __init__(self, phase: Phase, fold: int = None):
        super().__init__("models")
        self.phase = phase
        self.fold = fold
        if self.phase == Phase.CV and self.fold is None:
            raise ValueError("Fold must be specified for CV phase")
        if self.phase == Phase.BY_YEAR and self.fold is None:
            raise ValueError("Year must be specified for by-year phase")

    def get_subpath(self):
        if self.phase == Phase.CV or self.phase == Phase.BY_YEAR:
            return f'{self.phase}{"a" if self.fold == 0 else self.fold}'
        else:
            return f"{self.phase}"


class MetricsPath(OutputPath):
    def __init__(self, model: "BaseClassifier | str", phase: Phase, fold: int = None):
        super().__init__("metrics")
        if isinstance(model, str):
            self.model = model
        else:
            self.model = model.id
        self.phase = phase
        self.fold = fold
        if self.phase == Phase.CV and self.fold is None:
            raise ValueError("Fold must be specified for CV phase")
        if self.phase == Phase.BY_YEAR and self.fold is None:
            raise ValueError("Year must be specified for by-year phase")

    def get_subpath(self):
        if self.phase == Phase.CV or self.phase == Phase.BY_YEAR:
            return f'{self.model}-{self.phase}{"a" if self.fold == 0 else self.fold}'
        else:
            return f"{self.model}-{self.phase}"


class PredictionPath(OutputPath):
    def __init__(self, model: "BaseClassifier | str", phase: Phase, fold: int = None):
        super().__init__("predictions")
        if isinstance(model, str):
            self.model = model
        else:
            self.model = model.id
        self.phase = phase
        self.fold = fold
        if self.phase == Phase.CV and self.fold is None:
            raise ValueError("Fold must be specified for CV phase")
        if self.phase == Phase.BY_YEAR and self.fold is None:
            raise ValueError("Year must be specified for by-year phase")

    def get_subpath(self):
        if self.phase == Phase.CV or self.phase == Phase.BY_YEAR:
            return f'{self.model}-{self.phase}{"a" if self.fold == 0 else self.fold}'
        else:
            return f"{self.model}-{self.phase}"


class AnalysisPath(OutputPath):
    def __init__(self, phase: Phase, fold: int = None):
        super().__init__("analyses")
        self.phase = phase
        self.fold = fold
        if self.phase == Phase.CV and self.fold is None:
            raise ValueError("Fold must be specified for CV phase")
        if self.phase == Phase.BY_YEAR and self.fold is None:
            raise ValueError("Year must be specified for by-year phase")

    def get_subpath(self):
        if self.phase == Phase.CV or self.phase == Phase.BY_YEAR:
            return f'{self.phase}{"a" if self.fold == 0 else self.fold}'
        else:
            return f"{self.phase}"


class ModelAnalysisPath(AnalysisPath):
    def __init__(self, model: "BaseClassifier | str", phase: Phase, fold: int = None):
        super().__init__(phase, fold)
        if isinstance(model, str):
            self.model_id = model
            self.model = None
        else:
            self.model_id = model.id
            self.model = model

    def get_subpath(self):
        return f"{self.model_id}-{super().get_subpath()}"
