from tkinter import BooleanVar, IntVar, StringVar
from ...utils.param_classes.fix import FixsParams
from ...utils.param_classes.model_params.model import ModelParams
from ...utils.param_classes.save import SaveParams
from ...utils.param_classes.tracker import TrackerParams


class AutoLabellerParams:

    def __init__(self, data=None) -> None:
        if data is None:
            self.tracker = TrackerParams(None)
            self.save = SaveParams(None)
            self.fixs = FixsParams(None)
            return
        
        self.sort_files_with_time = BooleanVar(value=data["sort_files_with_time"])
        self.imgsz = IntVar(value=data["imgsz"])
        self.device = StringVar(value=data["device"])
        self.use_tracker = BooleanVar(value=data["use_tracker"])
        self.enable_save = BooleanVar(value=data["enable_save"])
        self.resize_img = BooleanVar(value=data["resize_img"])
        self.resize_height = IntVar(value=data["resize_height"])
        self.resize_width = IntVar(value=data["resize_width"])
        self.classes_txt = StringVar(value=data["classes_txt"])
        self.label_map_file = StringVar(value=data["label_map_file"])
        self.enable_label_map= BooleanVar(value=data["enable_label_map"])

        self.tracker = TrackerParams(data["tracker"])
        self.fixs = FixsParams(data["fixs"])
        self.save = SaveParams(data["save"])
        self.modelsParams=[ModelParams(x) for x in data["models_params"]]
    def addNewModel(self, file_path):
        modelParams=ModelParams(data=self.modelsParams[0].toJson())
        modelParams.model.set(file_path)
        modelParams.id=len(self.modelsParams)
        self.modelsParams.append(modelParams)
        return modelParams
    def removeModel(self, id):
        self.modelsParams=list(filter(lambda x: x.id!=id, self.modelsParams))
        return self.modelsParams

    def toJson(self):
        return {
            "sort_files_with_time":self.sort_files_with_time.get(),
            "enable_save": self.enable_save.get(),
            "use_tracker": self.use_tracker.get(),
            "imgsz": self.imgsz.get(),
            "resize_img": self.resize_img.get(),
            "resize_height": self.resize_height.get(),
            "resize_width": self.resize_width.get(),
            "device": self.device.get(),
            "classes_txt": self.classes_txt.get(),
            "label_map_file": self.label_map_file.get(),
            "enable_label_map": self.enable_label_map.get(),
            "tracker": self.tracker.toJson(),
            "save": self.save.toJson(),
            "models_params": [x.toJson() for x in self.modelsParams],
            "fixs": self.fixs.toJson()
        }
