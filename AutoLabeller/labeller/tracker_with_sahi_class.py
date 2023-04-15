from pathlib import Path
from numpy import asarray
from ..labeller.sahi_labeller import SahiLabeller
from ..trackers.strong_sort.strong_sort import StrongSORT
from ..trackers.multi_tracker_zoo import create_tracker
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import (Profile, check_img_size, cv2,
                                  strip_optimizer)
from yolov5.models.common import DetectMultiBackend
import torch
import numpy as np
import os
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, POSTPROCESS_NAME_TO_CLASS
from ..labeller.base_predictor import BasePredictor
from ..utils.image_resize import image_resize
from ..trackers.ocsort.ocsort import OCSort


from ..utils.params_saver import AutoLabellerParams, ParamsSaver
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


class Tracker(BasePredictor):
    frame = None
    prev_means = None
    prev_frame = []

    def __init__(self, params: AutoLabellerParams, update=False) -> None:
        self.update = update
        self.params = params
        self.labeller=SahiLabeller(params)
        self.model = self.labeller.models[0].model
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

        self.device = select_device(params.device.get())
        self.names = self.labeller.names
        self.imgsz = check_img_size(
            params.imgsz.get(), s=self.stride)  # check image size
        self.load_tracker()

    def load_tracker(self):

        nr_sources = 1
        self.vid_path, self.vid_writer, self.txt_path = [
            None] * nr_sources, [None] * nr_sources, [None] * nr_sources

        # Create as many strong sort instances as there are video sources
        self.tracker_list: list[StrongSORT | OCSort] = []
        for i in range(nr_sources):
            self.tracker = create_tracker(
                self.params.tracker.tracking_method.get(), Path(self.params.tracker.reid_weights.get()), self.device, self.params.tracker.half.get())
            self.tracker_list.append(self.tracker, )
            if hasattr(self.tracker_list[i], 'model'):
                if hasattr(self.tracker_list[i].model, 'warmup'):
                    self.tracker_list[i].model.warmup()
        self.outputs = [None] * nr_sources
        # Run tracking
        # model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
        self.seen, self.windows, self.dt = 0, [
        ], (Profile(), Profile(), Profile(), Profile())
        self.curr_frames, self.prev_frames = [
            None] * nr_sources, [None] * nr_sources

    def is_scene_changed(self, img: np.ndarray):
        def correlationCoefficient(X, Y):
            n = X.size
            sum_X = X.sum()
            sum_Y = Y.sum()
            sum_XY = (X*Y).sum()
            squareSum_X = (X*X).sum()
            squareSum_Y = (Y*Y).sum()
            corr = (n * sum_XY - sum_X * sum_Y)/(np.sqrt((n * squareSum_X -
                                                          sum_X * sum_X) * (n * squareSum_Y - sum_Y * sum_Y)))
            return corr


        if len(self.prev_frame) == 0:
            self.prev_frame = img
            return False

        cor = correlationCoefficient(img/255, self.prev_frame/255)
        self.prev_frame = img
        if cor <= self.params.tracker.min_correlation.get():
            print(cor)
            return True
        return False

    def detect(self, with_out_bbox_normalization=False):
        predictions = []
        cocos = []
        cocos =  self.labeller.get_prediction(self.frame, with_out_bbox_normalization=with_out_bbox_normalization)

        i = 0
        self.seen += 1
        self.curr_frames[i] = self.frame
        if hasattr(self.tracker_list[i], 'tracker') and hasattr(self.tracker_list[i].tracker, 'camera_update'):
            # camera motion compensation
            if self.prev_frames[i] is not None and self.curr_frames[i] is not None:
                self.tracker_list[i].tracker.camera_update(
                    self.prev_frames[i], self.curr_frames[i])

        if cocos is not None and len(cocos):
            self.outputs[i] = self.tracker_list[i].update_with_cocos(
                cocos, self.frame)

            # draw boxes for visualization
            if len(self.outputs[i]) > 0:
                for j, (x) in enumerate(self.outputs[i]):
                    bbox = x[0:4]
                    bbox[2] = x[2]-x[0]
                    bbox[3] = x[3]-x[1]
                    
                    if not with_out_bbox_normalization:
                        bbox=[a/(self.frame.shape[1] if i%2==0 else self.frame.shape[0]) for i, a in enumerate(bbox)]
                    predictions.append({'bbox': x[0:4], 'score':  x[6], 'category_id': int(x[5]),
                                        'category_name': self.names[int(x[5])], 'id': int(x[4])})

        self.prev_frames[i] = self.curr_frames[i]

        """ if self.update:
            # update model (to fix SourceChangeWarning)
            strip_optimizer(self.params.model.get()) """

        return predictions
    tracker_using_count = 0
    tracker_reset_count = 0

    @torch.no_grad()
    def get_prediction(self, img: str|np.ndarray):
        paramsaver=ParamsSaver()
        self.params=paramsaver.getParams()
        if type(img) is not np.ndarray:
            img=cv2.imread(img)
        if self.is_scene_changed(img):
            self.load_tracker()
            self.tracker_using_count = 0
            self.tracker_reset_count += 1
        print(f"tracker reset count: {self.tracker_reset_count}")
        self.frame=img
        cocos = self.detect()
        if self.tracker_using_count == 0 and len(cocos) == 0:
            cocos = self.detect()
        self.tracker_using_count += 1
        return cocos
