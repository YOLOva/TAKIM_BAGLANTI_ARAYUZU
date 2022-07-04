import time


if __name__ == "__main__":
    from sahi.model import Yolov5DetectionModel
    from sahi.utils.cv import read_image
    from sahi.predict import get_prediction, get_sliced_prediction, predict
    from sahi.utils.yolov5 import (
        download_yolov5s6_model,
    )
    import torchvision

    for_detect = "./src/yolova/test/3_284_dehazed_006840.png"
    detection_model = Yolov5DetectionModel(
        model_path="./src/yolova/models/best3.pt",
        image_size=640,
        confidence_threshold=0.5,
        device="cuda:0" #"cpu" 
    )
    
    t1 = time.perf_counter() 
    result = get_prediction(for_detect, detection_model)
    cocos = result.to_coco_annotations()
    print(cocos)
    t2 = time.perf_counter() 
    print(f"geçen süre, {t2-t1}")
    result.export_visuals(export_dir="demo_data/")

""" 
# Eğer 80. frame gelmişse geçen zamana bakılsın

import time
minute = 60
one_second=1
t1 = time.perf_counter()
for i in range(1, 170):
    time.sleep(one_second)
    t2 = time.perf_counter()
    if((t2 - t1) > 10*one_second):
        print(f"bekleme yapıldı {i}")
        print(f"geçen zaman {t2 - t1}")
        t1 = time.perf_counter()
print(f"geçen zaman {t2 - t1}") """