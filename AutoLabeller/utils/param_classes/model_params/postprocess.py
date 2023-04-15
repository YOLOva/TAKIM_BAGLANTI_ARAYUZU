from tkinter import BooleanVar, DoubleVar, StringVar


class PostProcessParams:
    def __init__(self, data=None) -> None:
        if data is None:
            return
        self.match_metric = StringVar(value=data["match_metric"])
        self.match_threshold = DoubleVar(value=data["match_threshold"])
        self.class_agnostic = BooleanVar(value=data["class_agnostic"])
        self.postprocess_type = StringVar(value=data["postprocess_type"])

    def toJson(self):
        return {
            "match_metric": self.match_metric.get(),
            "match_threshold": self.match_threshold.get(),
            "class_agnostic": self.class_agnostic.get(),
            "postprocess_type": self.postprocess_type.get()
        }
