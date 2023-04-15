from tkinter import BooleanVar


class UYZ2022FixsParams:
    def __init__(self, data=None) -> None:
        if data is None:
            return
        self.uaips_state_fix = BooleanVar(value=data["uaips_state_fix"])
        self.person_same_size_in_car_fix = BooleanVar(
            value=data["person_same_size_in_car_fix"])

    def toJson(self):
        return {
            "uaips_state_fix": self.uaips_state_fix.get(),
            "person_same_size_in_car_fix": self.person_same_size_in_car_fix.get()
        }
