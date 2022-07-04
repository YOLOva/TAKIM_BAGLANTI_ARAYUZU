import imp
import logging
import time
import numpy

import requests

from src.constants import classes, landing_statuses
from src.detected_object import DetectedObject

from src.yolova.constants import index_to_classes


from sahi.model import Yolov5DetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.utils.yolov5 import (
    download_yolov5s6_model,
)
import torchvision
import cv2

class ObjectDetectionModel:
    # Base class for team models
    use_sahi= True
    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        self.model = Yolov5DetectionModel(
            model_path="./src/yolova/models/best3.pt",
            image_size=640,
            confidence_threshold=0.5,
            device="cuda:0" # "cpu",
        )
        # Modelinizi bu kısımda init edebilirsiniz.
        # self.model = get_keras_model() # Örnektir!

    @staticmethod
    def download_image(index, img_url, images_folder):
        t1 = time.perf_counter()
        img_bytes = requests.get(img_url).content
        image_name = f'{index}_{img_url.split("/")[-1]}'  # frame_x.jpg
        image_path =images_folder + image_name
        with open(image_path, 'wb') as img_file:
            img_file.write(img_bytes)

        t2 = time.perf_counter()

        logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {image_path}')
        return image_path

    def process(self, index, prediction,evaluation_server_url):
        # Yarışmacılar resim indirme, pre ve post process vb işlemlerini burada gerçekleştirebilir.
        # Download image (Example)
        image_path = self.download_image(index,evaluation_server_url + "media" + prediction.image_url, "./_images/")
        # Örnek: Burada OpenCV gibi bir tool ile preprocessing işlemi yapılabilir. (Tercihe Bağlı)
        # ...
        # Nesne tespiti modelinin bulunduğu fonksiyonun (self.detect() ) çağırılması burada olmalıdır.
        frame_results = self.detect(prediction, image_path)
        # Tahminler objesi FramePrediction sınıfında return olarak dönülmelidir.
        return frame_results

    def detect(self, prediction, image_path):
        # Modelinizle bu fonksiyon içerisinde tahmin yapınız.
        # results = self.model.evaluate(...) # Örnektir.

        # Burada örnek olması amacıyla 20 adet tahmin yapıldığı simüle edilmiştir.
        # Yarışma esnasında modelin tahmin olarak ürettiği sonuçlar kullanılmalıdır.
        # Örneğin :
        # for i in results: # gibi
        #indirme hızı 0.8s
        if self.use_sahi: #1.9 - 2.1 saniye hız
            result = get_sliced_prediction(
                image_path,
                self.model,
                slice_height = 1024,
                slice_width = 1024,
                overlap_height_ratio = 0.1,
                overlap_width_ratio = 0.1
            )
        else: #1.5 - 1.6 saniye hız
            result = get_prediction(image_path, self.model)

        cocos = result.to_coco_annotations()
        detection_inis_group = []
        detection_diger_group = []
        for coco in cocos:
            bbox=coco["bbox"]

            cls = classes[index_to_classes[coco["category_id"]]],  # Tahmin edilen nesnenin sınıfı classes sözlüğü kullanılarak atanmalıdır.
            landing_status = landing_statuses["Inis Alani Degil"]  # Tahmin edilen nesnenin inilebilir durumu landing_statuses sözlüğü kullanılarak atanmalıdır.
            top_left_x = bbox[0]  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
            top_left_y = bbox[1]  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
            bottom_right_x = bbox[0]+bbox[2]  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
            bottom_right_y = bbox[1]+bbox[3]  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.

            # Modelin tespit ettiği herbir nesne için bir DetectedObject sınıfına ait nesne oluşturularak
            # tahmin modelinin sonuçları parametre olarak verilmelidir.
            d_obj = DetectedObject(cls,
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y)
            if(cls == 2 or cls == 3):
               d_obj.landing_status = landing_statuses["Inilebilir"]
               detection_inis_group.append(d_obj)
            else:
                detection_diger_group.append(d_obj)
            # Modelin tahmin ettiği her nesne prediction nesnesi içerisinde bulunan detected_objects listesine eklenmelidir.
        d_objs = self.inisAlaniKontrolu(detection_inis_group,detection_diger_group)
        for d_obj in d_objs:
            prediction.add_detected_object(d_obj)
        result.export_visuals(export_dir="last_detect/")

        cv2.imshow("process...", cv2.imread("last_detect\prediction_visual.png"))
        cv2.waitKey(10)
        return prediction

    def inisAlaniKontrolu(self,detection_inis_group, detection_diger_group):
        for detection_inis in detection_inis_group:
            for detection_diger in detection_diger_group:
                #sol üst köşe alanda mı
                if (
                    # cisim sol üst köşe koordinatları iniş alanında mı?
                    (
                        detection_inis.top_left_x <= detection_diger.top_left_x <= detection_inis.bottom_right_x
                        and
                        detection_inis.top_left_y <= detection_diger.top_left_y <= detection_inis.bottom_right_y
                    )
                    or
                    # ya da cisim sağ alt köşe koordinatları iniş alanında mı?
                    (
                        detection_inis.top_left_x <= detection_diger.bottom_right_x <= detection_inis.bottom_right_x
                        and
                        detection_inis.top_left_y <= detection_diger.bottom_right_y <= detection_inis.bottom_right_y
                    )
                    or
                    # cisim sağ üst köşe koordinatları iniş alanında mı?
                    (
                        detection_inis.top_left_x <= detection_diger.bottom_right_x <= detection_inis.bottom_right_x
                        and
                        detection_inis.top_left_y <= detection_diger.top_left_y <= detection_inis.bottom_right_y
                    )
                    or
                    # cisim sol alt köşe koordinatları iniş alanında mı?
                    (
                        detection_inis.top_left_x <= detection_diger.top_left_x <= detection_inis.bottom_right_x
                        and
                        detection_inis.top_left_y <= detection_diger.bottom_right_y <= detection_inis.bottom_right_y
                    )
                ):
                    detection_inis.landing_status = landing_statuses["Inilemez"]
                    break
            detection_diger_group.append(detection_inis)
        return detection_diger_group