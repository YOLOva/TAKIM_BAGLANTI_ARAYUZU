from tkinter import BooleanVar, DoubleVar, IntVar, StringVar
from ....utils.param_classes.model_params.postprocess import PostProcessParams
from ....utils.param_classes.model_params.sahi import SahiParams
from ....utils.param_classes.tracker import TrackerParams


class ModelParams:
    def __init__(self, data=None) -> None:
        if data is None:
            self.sahi = SahiParams(None)
            self.tracker = TrackerParams(None)
            self.postprocess = PostProcessParams(None)
            return
        self.data = data
        self.id=StringVar(value=data["id"])
        self.model = StringVar(value=data["model"])
        self.name = StringVar(value=data["name"])

        self.conf= DoubleVar(value=data["conf"])
        self.verbose = IntVar(value=data["verbose"])
        self.use_sahi = BooleanVar(value=data["use_sahi"])
        self.sahi = SahiParams(data["sahi"])
        self.postprocess = PostProcessParams(data["postprocess"])

    def toJson(self):
        return {
            "use_sahi": self.use_sahi.get(),
            "conf": self.conf.get(),
            "model": self.model.get(),
            "id": self.model.get(),
            "name": self.name.get(),
            "verbose": self.verbose.get(),
            "postprocess": self.postprocess.toJson(),
            "sahi": self.sahi.toJson()
        }
