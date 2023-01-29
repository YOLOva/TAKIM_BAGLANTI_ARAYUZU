import logging
import time
import requests
from src.AutoLabeller.auto_labeller import AutoLabeller
from src.frame_predictions import FramePredictions
from src.detected_object import DetectedObject
from pathlib import Path


class ObjectDetectionModel:
    current_video_name=""
    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        Path("./_labels/").mkdir(exist_ok=True, parents=False)
        self.confidences = [0.4, 0.4, 0.4, 0.4]
        self.generate_labeller()

    def generate_labeller(self):
        self.labeller = AutoLabeller(yolo_weights=r"src\AutoLabeller\YOLOva2022Best.pt", device="cuda:0", labels_output_folder="./_labels/",
                                show_vid=False, conf_thres=min(self.confidences), check_inilebilir=True, label_mapper=r"src\AutoLabeller\4class.txt", classes_txt=r"src\AutoLabeller\4classes.txt")  # "cpu", # or 'cuda:0'
    def download_image(self, index, img_url, images_folder):
        t1 = time.perf_counter()
        img_bytes = requests.get(img_url).content
        image_name = f'{index}_{img_url.split("/")[-1]}'  # frame_x.jpg
        image_path = images_folder + image_name
        with open(image_path, 'wb') as img_file:
            img_file.write(img_bytes)
        t2 = time.perf_counter()
        logging.info(
            f'{img_url} - Download Finished in {t2 - t1} seconds to {image_path}')
        download_time = t2 - t1
        return (image_path, download_time)

    def process(self, index, prediction:FramePredictions, evaluation_server_url):
        (image_path, download_time) = self.download_image(
            index, evaluation_server_url + "media" + prediction.image_url, "./_images/")
        prediction.image_path=image_path
        prediction.download_time=download_time
        t1 = time.perf_counter()
        frame_results = self.detect(prediction, image_path)
        t2 = time.perf_counter()
        prediction.detection_time=t2-t1
        prediction.names=self.labeller.names
        return frame_results

    def detect(self, prediction:FramePredictions, image_path):
        if self.current_video_name !="" and self.current_video_name!=prediction.video_name:
            self.current_video_name=prediction.video_name
            self.generate_labeller()
        cocos=self.labeller.detect(source=image_path)
        prediction.cocos=cocos
        for coco in cocos:
            score = coco["score"]
            if coco["category_id"] is tuple:
                cls=coco["category_id"][0]
            else:
                cls = coco["category_id"],
            if (self.confidences[cls[0]] > score):
                continue
            bbox = coco["bbox"]
            landing_status = coco["inilebilir"]
            top_left_x = bbox[0]
            top_left_y = bbox[1]
            bottom_right_x = bbox[0]+bbox[2]
            bottom_right_y = bbox[1]+bbox[3]
            
            d_obj = DetectedObject(cls[0],
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y)
            prediction.add_detected_object(d_obj)
        return prediction
