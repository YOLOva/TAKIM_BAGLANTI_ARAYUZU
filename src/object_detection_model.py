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

from pathlib import Path
class ObjectDetectionModel:
    # Base class for team models
    use_sahi= True
    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        self.model = Yolov5DetectionModel(
            model_path="./src/yolova/models/best.pt",
            image_size=1088,
            confidence_threshold=0.4,
            device="cuda:0" # "cpu",
        )
        self.confidences = [0.47, 0.4, 0.4, 0.4]
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
        
        t1 = time.perf_counter() # başlangıç zamanı
        frame_results = self.detect(prediction, image_path)
        # Tahminler objesi FramePrediction sınıfında return olarak dönülmelidir.
        t2 = time.perf_counter() # başlangıç zamanı
        print("Tespit Süresi", t2-t1)
        return frame_results

    def detect(self, prediction, image_path):
        if self.use_sahi: #1.9 - 2.1 saniye hız
            result = get_sliced_prediction(
                image_path,
                self.model,
                slice_height = 1600,
                slice_width = 1600,
                overlap_height_ratio = 0.9,
                overlap_width_ratio = 0.9
            )
        else: #1.5 - 1.6 saniye hız
            result = get_prediction(image_path, self.model)

        cocos = result.to_coco_annotations()
        detection_inis_group = []
        detection_diger_group = []
        for coco in cocos:
            score = coco["score"]
            cls = classes[index_to_classes[coco["category_id"]]],
            if(self.confidences[cls[0]]>score):
                continue  # Tahmin edilen nesnenin sınıfı classes sözlüğü kullanılarak atanmalıdır.
            bbox=coco["bbox"]
            landing_status = landing_statuses["Inis Alani Degil"]  # Tahmin edilen nesnenin inilebilir durumu landing_statuses sözlüğü kullanılarak atanmalıdır.
            top_left_x = bbox[0]  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
            top_left_y = bbox[1]  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
            bottom_right_x = bbox[0]+bbox[2]  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
            bottom_right_y = bbox[1]+bbox[3]  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
            # Modelin tespit ettiği herbir nesne için bir DetectedObject sınıfına ait nesne oluşturularak
            # tahmin modelinin sonuçları parametre olarak verilmelidir.
            d_obj = DetectedObject(cls[0],
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y)
            if(cls[0] in [2,3]):
               d_obj.landing_status = landing_statuses["Inilebilir"]
               detection_inis_group.append(d_obj)
            else:
                detection_diger_group.append(d_obj)
            # Modelin tahmin ettiği her nesne prediction nesnesi içerisinde bulunan detected_objects listesine eklenmelidir.
        d_objs = self.inisAlaniKontrolu(detection_inis_group,detection_diger_group)
        d_objs = self.arac_insan_fix(d_objs)
        for d_obj in d_objs:
            prediction.add_detected_object(d_obj)
        image_name = Path(image_path).stem
        result.export_visuals(export_dir="last_detect/", file_name=image_name)

        cv2.imshow("process...", cv2.imread(f"last_detect\{image_name}.png"))
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
    def arac_insan_fix(self, detects):
        araclar = [x for x in detects if x.cls == 0]
        insanlar = [x for x in detects if x.cls == 1]
        diger = [x for x in detects if x.cls in [2,3]]
        for insan in insanlar:
            insan_w = abs(insan.top_left_x - insan.bottom_right_x)
            for arac in araclar:
                arac_w = abs(arac.top_left_x - arac.bottom_right_x)
                ratio_w = insan_w/arac_w
                if (# cisim sol üst köşe koordinatları iniş alanında mı?
                (
                    insan.top_left_x <= arac.top_left_x <= insan.bottom_right_x
                    and
                    insan.top_left_y <= arac.top_left_y <= insan.bottom_right_y
                )
                or
                # ya da cisim sağ alt köşe koordinatları iniş alanında mı?
                (
                    insan.top_left_x <= arac.bottom_right_x <= insan.bottom_right_x
                    and
                    insan.top_left_y <= arac.bottom_right_y <= insan.bottom_right_y
                )
                or
                # cisim sağ üst köşe koordinatları iniş alanında mı?
                (
                    insan.top_left_x <= arac.bottom_right_x <= insan.bottom_right_x
                    and
                    insan.top_left_y <= arac.top_left_y <= insan.bottom_right_y
                )
                or
                # cisim sol alt köşe koordinatları iniş alanında mı?
                (
                    insan.top_left_x <= arac.top_left_x <= insan.bottom_right_x
                    and
                    insan.top_left_y <= arac.bottom_right_y <= insan.bottom_right_y
                )):
                    if (ratio_w>=0.8):
                        insanlar.remove(insan)
                        
        return araclar + insanlar + diger