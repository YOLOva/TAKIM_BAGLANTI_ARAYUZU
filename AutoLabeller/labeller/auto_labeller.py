

import cv2
import numpy as np
import torch
from .fix.custom.custom_fix import CustomFix

from AutoLabeller.root import get_root
from ..labeller.fix.arac_insan import AracInsanFix
from ..labeller.fix.fix import AllClassesFixs
from .fix.custom.uaips import UAIPFix
from ..labeller.sahi_labeller import SahiLabeller
from ..labeller.tracker_with_sahi_class import Tracker
import os
from PIL import Image
from pathlib import Path
import shutil
import statistics
from ..utils.image_resize import image_resize
from ..utils.params_saver import ParamsSaver

from ..utils.yolo_annotation import YoloAnnotation


class AutoLabeller:
    def __init__(self,
                 labels_output_folder="") -> None:
        torch.cuda.empty_cache()
        self.params_saver=ParamsSaver()
        self.params=self.params_saver.getParams()
        if self.params.use_tracker.get():
            self.predictor = Tracker(self.params)
        else:
            self.predictor = SahiLabeller(self.params)
        self.names = self.predictor.names
        self.labels_output_folder = labels_output_folder
        self.detect_index = 0

        self.class_map = {}
        if self.params.label_map_file.get() != "" and self.params.enable_label_map.get():
            with open(os.path.join(get_root(),self.params.label_map_file.get()), "r") as file:
                lines = file.readlines()
            for i, line in enumerate(lines):
                self.class_map[i]={
                    "name":line.split()[1],
                    "id":int(line.split()[0])
                }
            print(self.class_map)

        if len(self.class_map) == 0:
            self.class_map = {i:{"id":i, "name":name} for i, name in enumerate(self.names)}

    def detect(self,source, img:np.ndarray=None, save_txt=True, info=None):
        if type(img) is None:
            img=cv2.imread(source)
        cocos = self.predictor.get_prediction(img)
        cocos = self.fix(cocos)
        cocos = self.map_class_ids(cocos)
        if save_txt:
            self.write_labels(cocos, source)
        self.detect_index += 1
        return cocos
    
    def map_class_ids(self, cocos):
        for coco in cocos:
            cls = coco["category_id"]
            name=self.class_map[cls]['name']
            class_id = self.class_map[cls]['id']
            if(class_id==-1): continue
            if self.params.fixs.uyz2022.uaips_state_fix.get() and cls in [2, 3] and coco["inilebilir"] == 0:
                class_id = class_id+1
            coco["category_id"]=class_id
            coco["category_name"]=name
        return cocos

    def write_labels(self, cocos, source):
        txt_path = str(Path(self.labels_output_folder))
        Path(txt_path).mkdir(parents=True, exist_ok=True)
        classes_path=os.path.join(
            txt_path, "classes.txt")
        if not os.path.exists(classes_path):
            shutil.copyfile(os.path.join(get_root(),self.params.classes_txt.get()),classes_path)

        yolo_annotation = YoloAnnotation(
            f"{txt_path}\\{Path(source).stem}.txt")   
        yolo_annotation.write_cocos(cocos)

    def fix(self, cocos):
        # TODO: İnsanların belli bir boyuttan büyük olması temizlenme aşamasına girmeli
        for other in [coco for coco in cocos if coco["category_id"] not in [2, 3]]:
            other["inilebilir"] = -1
        for other in [coco for coco in cocos if coco["category_id"] in [2, 3]]:
            other["inilebilir"] = 1
        allClassesFixs=AllClassesFixs(self.params, self.predictor.frame)
        uAIPFix=UAIPFix(self.predictor.frame, allClassesFixs)
        aracInsanFix=AracInsanFix()
        if self.params.fixs.negative_bbox_values_fix.get():
            cocos = allClassesFixs.negative_value_fix(cocos)
        if self.params.fixs.same_size_class_box_in_box_fix.get():
            cocos = allClassesFixs.multiple_box_in_box_fix(cocos)
        if self.params.fixs.enable_uyz2022_fix.get():
            if self.params.fixs.uyz2022.uaips_state_fix.get():
                cocos = uAIPFix.fix(cocos)
            if self.params.fixs.uyz2022.person_same_size_in_car_fix.get():
                cocos = aracInsanFix.fix(cocos)
        custom_fix=CustomFix(self.predictor.frame)
        cocos=custom_fix.fix(cocos)
        return cocos


    

    
