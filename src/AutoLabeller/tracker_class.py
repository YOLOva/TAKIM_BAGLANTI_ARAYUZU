import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

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

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_img_size, non_max_suppression, scale_boxes, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.segment.general import masks2segments, process_mask, process_mask_native
from trackers.multi_tracker_zoo import create_tracker

class Tracker():
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
                 ) -> None:
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

        # Load model
        device = select_device(device)
        self.model = DetectMultiBackend(
            yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        nr_sources = 1
        self.vid_path, self.vid_writer, self.txt_path = [
            None] * nr_sources, [None] * nr_sources, [None] * nr_sources

        # Create as many strong sort instances as there are video sources
        self.tracker_list = []
        for i in range(nr_sources):
            self.tracker = create_tracker(
                tracking_method, reid_weights, device, half)
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

    @torch.no_grad()
    def get_prediction(self, source: str):
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

                """ if hasattr(self.tracker_list[i], 'tracker') and hasattr(self.tracker_list[i].tracker, 'camera_update'):
                    # camera motion compensation
                    if self.prev_frames[i] is not None and self.curr_frames[i] is not None:
                        self.tracker_list[i].tracker.camera_update(
                            self.prev_frames[i], self.curr_frames[i]) """

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


