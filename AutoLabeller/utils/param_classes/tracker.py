from pathlib import Path
from tkinter import BooleanVar, DoubleVar, StringVar

from ...utils.param_classes.model_params.strong_sort import StrongSortParams


class TrackerParams:
    def __init__(self, data=None) -> None:
        if data is None:
            self.strongsort = StrongSortParams(None)
            return
        self.min_correlation = DoubleVar(value=data["min_correlation"])
        self.half = BooleanVar(value=data["half"])
        self.tracking_method = StringVar(value=data["tracking_method"])
        self.reid_weights = StringVar(value=data["reid_weights"])
        self.strongsort = StrongSortParams(data["strongsort"])

    def toJson(self):
        return {
            "min_correlation": self.min_correlation.get(),
            "half": self.half.get(),
            "tracking_method": self.tracking_method.get(),
            "reid_weights": self.reid_weights.get(),
            "strongsort": self.strongsort.toJson()
        }
