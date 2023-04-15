import os
from pathlib import Path
import yaml

from AutoLabeller.root import get_root
from ..utils.param_classes.auto_labeller import AutoLabellerParams

class ParamsSaver:
    def __init__(self) -> None:
        self.param_file = os.path.join(str(get_root()), "data/saved.yaml")
        self.default_file = os.path.join(str(get_root()), "data/default.yaml")

    def get_defaults(self):
        return self.get_data(self.default_file)

    def get_data(self, conf_file=None):
        conf_file = self.param_file if conf_file is None else conf_file
        fo = open(conf_file, 'r')
        return yaml.load(fo.read(), Loader=yaml.FullLoader)

    def getParams(self):
        return AutoLabellerParams(data=self.get_data())

    def getDefaultParams(self):
        return AutoLabellerParams(data=self.get_defaults())
    
    def saveParams(self, params:AutoLabellerParams):
        self.save_data(params.toJson())

    def save_data(self, data):
        fo = open(self.param_file, 'w')
        yaml.dump(data, fo, default_flow_style=False)

    def restore_defaults(self):
        data = self.get_defaults()
        self.save_data(data)
