import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, POSTPROCESS_NAME_TO_CLASS
from sahi.models.yolov5 import Yolov5DetectionModel
from sahi.prediction import PredictionResult
import cv2

from ..labeller.base_predictor import BasePredictor
from ..utils.params_saver import AutoLabellerParams, ParamsSaver
class SahiLabeller(BasePredictor):
    frame=None
    def __init__(self, params:AutoLabellerParams
                 ) -> None:
        self.models:list[Yolov5DetectionModel]=[]
        self.names=[]
        for modelParams in params.modelsParams:
            model:Yolov5DetectionModel = AutoDetectionModel.from_pretrained(
                model_type='yolov5',
                model_path=modelParams.model.get(),
                confidence_threshold=modelParams.conf.get(),
                image_size=modelParams.imgsz.get(),
                device=params.device.get()  # "cpu", # or 'cuda:0'
            )
            model.agnostic=modelParams.postprocess.model_class_agnostic.get()
            self.names+=model.category_names
            self.models.append(model)
        self.params=params


    def get_prediction(self, img: str|np.ndarray, with_out_bbox_normalization=False)->list:
        paramsaver=ParamsSaver()
        self.params=paramsaver.getParams()
        if type(img) is not np.ndarray:
            img=cv2.imread(img)
        self.frame=img
        cocos=[]
        curr_length=0
        for i, model in enumerate(self. models):
            modelParams=self.params.modelsParams[i]
            model.agnostic=modelParams.postprocess.model_class_agnostic.get()
            model.confidence_threshold=modelParams.conf.get()
            model.mask_threshold=modelParams.postprocess.match_threshold.get()
            if modelParams.use_sahi.get():
                try:
                    new_result = get_sliced_prediction(
                        self.frame,
                        model,
                        slice_height=modelParams.sahi.slice_height.get() if modelParams.sahi.auto_slice_resolution.get() == False else None,
                        slice_width=modelParams.sahi.slice_width.get() if modelParams.sahi.auto_slice_resolution.get() == False else None,
                        overlap_height_ratio=modelParams.sahi.overlap_height_ratio.get(),
                        overlap_width_ratio=modelParams.sahi.overlap_width_ratio.get(),
                        postprocess_type=modelParams.postprocess.postprocess_type.get(),
                        postprocess_match_metric=modelParams.postprocess.match_metric.get(),
                        postprocess_match_threshold=modelParams.postprocess.match_threshold.get(),
                        postprocess_class_agnostic=modelParams.postprocess.sahi_class_agnostic.get(),
                        verbose=modelParams.verbose.get(),
                        auto_slice_resolution=modelParams.sahi.auto_slice_resolution.get(),
                        perform_standard_pred=modelParams.sahi.perform_standard_pred.get()
                    )
                except IndexError as e:
                    print(e)
            else:
                postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[modelParams.postprocess.postprocess_type.get()]
                try:
                    new_result = get_prediction(self.frame, model, postprocess= postprocess_constructor(
                        match_threshold=modelParams.postprocess.match_threshold.get(), match_metric=modelParams.postprocess.match_metric.get(), class_agnostic=modelParams.postprocess.sahi_class_agnostic.get()), verbose=modelParams.verbose.get())
                except IndexError as e:
                    print(e)
            if new_result is not None:
                cur_cocos=new_result.to_coco_predictions()
                if(i==0): 
                    cocos=cur_cocos 
                else:
                    for coco in cur_cocos:
                        coco["category_id"]+=curr_length
                    cocos+=cur_cocos
            curr_length+=len(model.category_names)
            
        for i, coco in enumerate(cocos):
            coco["id"]=i
            if not with_out_bbox_normalization:
                coco["bbox"]=[a/(self.frame.shape[1] if i%2==0 else self.frame.shape[0]) for i, a in enumerate(coco["bbox"])]
        return cocos