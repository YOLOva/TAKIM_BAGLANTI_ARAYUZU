
from tkinter import BooleanVar, DoubleVar, IntVar


class SahiParams:
    def __init__(self, data) -> None:
        if data is None:
            return
        self.slice_height = IntVar(value=data["slice_height"])
        self.slice_width = IntVar(value=data["slice_width"])
        self.overlap_height_ratio = DoubleVar(
            value=data["overlap_height_ratio"])
        self.overlap_width_ratio = DoubleVar(value=data["overlap_width_ratio"])
        self.auto_slice_resolution = BooleanVar(
            value=data["auto_slice_resolution"])
        self.perform_standard_pred=BooleanVar(value=data["perform_standard_pred"])

    def toJson(self):
        return {
            "slice_height": self.slice_height.get(),
            "slice_width": self.slice_width.get(),
            "overlap_height_ratio": self.overlap_height_ratio.get(),
            "overlap_width_ratio": self.overlap_width_ratio.get(),
            "auto_slice_resolution": self.auto_slice_resolution.get(),
            "perform_standard_pred": self.perform_standard_pred.get()
        }
