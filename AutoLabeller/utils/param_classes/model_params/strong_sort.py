
from tkinter import DoubleVar, IntVar


class StrongSortParams:
    def __init__(self, data) -> None:
        if data is None:
            return
        self.data = data
        self.max_dist = DoubleVar(value=self.data["max_dist"])
        self.max_iou_distance = DoubleVar(value=self.data["max_iou_distance"])
        self.max_age = IntVar(value=self.data["max_age"])
        self.max_unmatched_preds = IntVar(
            value=self.data["max_unmatched_preds"])
        self.n_init = IntVar(value=self.data["n_init"])
        self.nn_budget = IntVar(value=self.data["nn_budget"])
        self.mc_lambda = DoubleVar(value=self.data["mc_lambda"])
        self.ema_alpha = DoubleVar(value=self.data["ema_alpha"])

    def toJson(self):
        return {
            "mc_lambda": self.mc_lambda.get(),
            "ema_alpha": self.ema_alpha.get(),
            "max_dist": self.max_dist.get(),
            "max_iou_distance": self.max_iou_distance.get(),
            "max_unmatched_preds": self.max_unmatched_preds.get(),
            "max_age": self.max_age.get(),
            "n_init": self.n_init.get(),
            "nn_budget": self.nn_budget.get()
        }
