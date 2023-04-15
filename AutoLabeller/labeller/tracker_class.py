
import os

from ..labeller.base_predictor import BasePredictor
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages
from yolov5.utils.general import (Profile, check_img_size, non_max_suppression, scale_boxes, cv2,
                                  strip_optimizer, check_file)
from yolov5.utils.torch_utils import select_device
from ..trackers.multi_tracker_zoo import create_tracker

from ..trackers.strong_sort.strong_sort import StrongSORT

from numpy import asarray
class Tracker(BasePredictor):
    frame=None
    prev_means=None
    prev_frame=[]
    def __init__(self,
                 yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
                 reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
                 tracking_method='strongsort',
                 imgsz=(640, 640),  # inference size (height, width)
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 save_trajectories=False,  # save trajectories for each track
                 nosave=False,  # do not save images/videos
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 update=False,  # update all models
                 half=False,  # use FP16 half-precision inference
                 dnn=False,  # use OpenCV DNN for ONNX inference
                 min_correlationn=0.68
                 ) -> None:
        self.min_correlationn=min_correlationn
        self.nosave = nosave
        self.device = device
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.half = half
        self.save_trajectories = save_trajectories
        self.tracking_method = tracking_method
        self.update = update
        self.yolo_weights = yolo_weights
        self.reid_weights=reid_weights
        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(
            self.yolo_weights, device=self.device, dnn=dnn, data=None, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.load_tracker()
        
    def load_tracker(self):

        nr_sources = 1
        self.vid_path, self.vid_writer, self.txt_path = [
            None] * nr_sources, [None] * nr_sources, [None] * nr_sources

        # Create as many strong sort instances as there are video sources
        self.tracker_list:list[StrongSORT] = []
        for i in range(nr_sources):
            self.tracker = create_tracker(
                self.tracking_method, self.reid_weights, self.device, self.half)
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
    
    def is_scene_changed(self, source:str):
        def correlationCoefficient(X, Y):
            n = X.size
            sum_X = X.sum()
            sum_Y = Y.sum()
            sum_XY = (X*Y).sum()
            squareSum_X = (X*X).sum()
            squareSum_Y = (Y*Y).sum()
            corr = (n * sum_XY - sum_X * sum_Y)/(np.sqrt((n * squareSum_X - sum_X * sum_X)* (n * squareSum_Y - sum_Y * sum_Y))) 
            return corr
        
        frame=asarray(cv2.imread(source))

        if len(self.prev_frame)==0:
            self.prev_frame=frame
            return False
        
        cor =correlationCoefficient(frame/255, self.prev_frame/255)
        self.prev_frame=frame
        if cor<=self.min_correlationn:
            print(cor)
            return True
        return False

       
        """  frame=cv2.imread(source)
        arr=asarray(frame)
        (B, G, R) = cv2.split(arr)
        means=[np.mean(B), np.mean(G), np.mean(R)]
        if self.prev_means==None:
            self.prev_means=means
            return False
        for i, mean in enumerate(means):
            change_ratio=abs(mean-self.prev_means[i])/mean
            if change_ratio>=self.rgb_mean_change_ratio:
                return True
        mean_p=np.mean(self.prev_means)
        mean_c=np.mean(means)
        change_ratio=abs(mean_p-mean_c)/mean_c
        self.prev_means=means
        if change_ratio>=self.rgb_mean_change_ratio:   
            return True
        return False """

    def detect(self, source):
        predictions = []
        # Source Detection
        source = str(Path(source))
        is_file = Path(source).suffix[1:] in (VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        if is_url and is_file:
            source = check_file(source)  # download

        # load image source
        dataset = LoadImages(source, img_size=self.imgsz,
                             stride=self.stride, auto=self.pt)

        for (path, im, im0s, vid_cap, s) in dataset:
            with self.dt[0]:
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.half else im.float()  # uint8 to fp16/32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            
            # Inference
            with self.dt[1]:
                pred = self.model(im, augment=self.augment,
                                  visualize=False)

            # Apply NMS
            with self.dt[2]:
                pred = non_max_suppression(
                    pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
              # Process detections
            for i, det in enumerate(pred):  # detections per image
                self.seen += 1
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                im_h, im_w, _ = im0.shape
                self.curr_frames[i] = im0
                self.frame=im0
                if hasattr(self.tracker_list[i], 'tracker') and hasattr(self.tracker_list[i].tracker, 'camera_update'):
                    # camera motion compensation
                    if self.prev_frames[i] is not None and self.curr_frames[i] is not None:
                        self.tracker_list[i].tracker.camera_update(
                            self.prev_frames[i], self.curr_frames[i])

                if det is not None and len(det):
                    # rescale boxes to im0 size
                    det[:, :4] = scale_boxes(
                        im.shape[2:], det[:, :4], im0.shape).round()

                    # pass detections to strongsort
                    with self.dt[3]:
                        self.outputs[i] = self.tracker_list[i].update(
                            det.cpu(), im0)
                    # draw boxes for visualization
                    if len(self.outputs[i]) > 0:
                        for j, (x) in enumerate(self.outputs[i]):
                            bbox=x[0:4]
                            bbox[2]=x[2]-x[0]
                            bbox[3]=x[3]-x[1]
                            predictions.append({'bbox': x[0:4], 'score':  x[6], 'category_id': int(x[5]),
                                'category_name': self.names[int(x[5])], 'id': int(x[4])})

                self.prev_frames[i] = self.curr_frames[i]

        if self.update:
            # update model (to fix SourceChangeWarning)
            strip_optimizer(self.yolo_weights)
        
        """ result = list(map(lambda x: {'bbox': x[0][0][0:4], 'score': x[0][0][6], 'category_id': x[0][0][5],
                    'category_name': self.names[x[0][0][5]], 'id': x[0][0][4]}, self.outputs)) """
        return predictions
    tracker_using_count=0
    tracker_reset_count=0
    @torch.no_grad()
    def get_prediction(self, source: str):
        if self.is_scene_changed(source):
            self.load_tracker()
            self.tracker_using_count=0
            self.tracker_reset_count+=1
        print(f"tracker reset count: {self.tracker_reset_count}")
        
        cocos= self.detect(source)
        if self.tracker_using_count==0 and len(cocos)==0:
            cocos= self.detect(source)
        self.tracker_using_count+=1
        return cocos


