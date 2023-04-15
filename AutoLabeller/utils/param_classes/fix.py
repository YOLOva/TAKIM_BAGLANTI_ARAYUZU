

from tkinter import BooleanVar, DoubleVar
from ...utils.param_classes.model_params.uyz_2022 import UYZ2022FixsParams


class FixsParams:
    uyz2022:UYZ2022FixsParams=None
    def __init__(self, data=None) -> None:
        if data is None:
            self.uyz2022=UYZ2022FixsParams()
            return
        self.enable_uyz2022_fix=BooleanVar(value=data["enable_uyz2022_fix"])
        self.negative_bbox_values_fix=BooleanVar(value=data["negative_bbox_values_fix"])
        self.same_size_class_box_in_box_fix=BooleanVar(value=data["same_size_class_box_in_box_fix"])
        self.same_size_class_box_in_box_fix_ratio= DoubleVar(value=data["same_size_class_box_in_box_fix_ratio"])
        self.uyz2022=UYZ2022FixsParams(data["uyz2022"])
    def toJson(self):
        return {
            "negative_bbox_values_fix": self.negative_bbox_values_fix.get(),
            "same_size_class_box_in_box_fix": self.same_size_class_box_in_box_fix.get(),
            "same_size_class_box_in_box_fix_ratio":self.same_size_class_box_in_box_fix_ratio.get(),
            "enable_uyz2022_fix": self.enable_uyz2022_fix.get(),
            "uyz2022":self.uyz2022.toJson()
        }
