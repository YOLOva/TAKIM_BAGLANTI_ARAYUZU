import logging
import os
import time
import cv2
import numpy as np
import requests
from AutoLabeller.labeller.auto_labeller import AutoLabeller
from AutoLabeller.utils.image_resize import image_resize
from AutoLabeller.utils.params_saver import ParamsSaver
from root import get_root
from src.frame_predictions import FramePredictions
from src.detected_object import DetectedObject
from pathlib import Path

from src.yolova.constants import classes_map

class ObjectDetectionModel:
    current_video_name=""
    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        Path("./_labels/").mkdir(exist_ok=True, parents=False)
        #self.confidences = [0.4, 0.4, 0.4, 0.4]

    def generate_labeller(self, output_folder):
        label_mapper = r"src\AutoLabeller\4class.txt"
        classes_txt = r"src\AutoLabeller\4classes.txt"
        
        self.labeller = AutoLabeller(output_folder)  # "cpu", # or 'cuda:0'
    def download_image(self, index, img_url, images_folder):
        t1 = time.perf_counter()
        img_bytes = requests.get(img_url).content
        image_name = f'{img_url.split("/")[-1]}'  # frame_x.jpg
        image_path = images_folder + image_name
        Path(images_folder).mkdir(parents=True, exist_ok=True)
        with open(image_path, 'wb') as img_file:
            img_file.write(img_bytes)
        t2 = time.perf_counter()
        logging.info(
            f'{img_url} - Download Finished in {t2 - t1} seconds to {image_path}')
        download_time = t2 - t1
        return (image_path, download_time)

    def process(self, index, prediction:FramePredictions, evaluation_server_url):
        output_folder = f"_output/session_{prediction.session}/{prediction.video_name}"
        if self.current_video_name =="" or self.current_video_name!=prediction.video_name:
            self.current_video_name=prediction.video_name
            self.generate_labeller(os.path.join(get_root(),f"{output_folder}/labels/"))
        (image_path, download_time) = self.download_image(
            index, evaluation_server_url + "media" + prediction.image_url, f"{output_folder}/images/")
        prediction.image_path=image_path
        prediction.download_time=download_time
        t1 = time.perf_counter()
        frame_results = self.detect(prediction, image_path)
        t2 = time.perf_counter()
        prediction.detection_time=t2-t1
        prediction.names=self.labeller.names
        return frame_results

    def detect(self, prediction:FramePredictions, image_path):
        paramsaver=ParamsSaver()
        self.params=paramsaver.getParams()
        img=cv2.imread(image_path)
        h,w=img.shape[:2]
        if self.params.resize_img.get() and img.shape[0] > self.params.resize_height.get():
            img = image_resize(img, height=self.params.resize_height.get(), width=self.params.resize_width.get())
        cocos=self.labeller.detect(image_path, img, True)
        prediction.cocos=cocos
        for coco in cocos:
            score = coco["score"]
            if isinstance(coco["category_id"], tuple):
                cls=coco["category_id"][0]
            else:
                cls = coco["category_id"]
            
            map_class=classes_map[cls]
            cls=map_class["id"]
            
            """ if (self.confidences[cls[0]] > score):
                continue """
            bbox = coco["bbox"]
            landing_status = map_class["inilebilir"]
            top_left_x = bbox[0]*w
            top_left_y = bbox[1]*h
            bottom_right_x = (bbox[0]+bbox[2])*w
            bottom_right_y = (bbox[1]+bbox[3])*h
            coco["inilebilir"]=landing_status
            d_obj = DetectedObject(cls,
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y)
            prediction.add_detected_object(d_obj)
        return prediction
