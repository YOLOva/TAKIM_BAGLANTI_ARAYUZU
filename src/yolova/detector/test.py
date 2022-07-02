"""if __name__ == "__main__":
    from sahi.model import Yolov5DetectionModel
    from sahi.utils.cv import read_image
    from sahi.predict import get_prediction, get_sliced_prediction, predict
    from IPython.display import Image
    from sahi.utils.yolov5 import (
        download_yolov5s6_model,
    )
    import torchvision

    for_detect = ".\test\2_35_frame_000788.jpg"
    detection_model = Yolov5DetectionModel(
        model_path="D:\Teknofest\YOLOVA\ConnectionInterface\TAKIM_BAGLANTI_ARAYUZU\src\yolova\models\Best3_Ekleme9.pt",
        image_size=640,
        confidence_threshold=0.5,
        device="cuda:0" # "cpu",
    )
    
    result = get_prediction(for_detect, detection_model)
    result.export_visuals(export_dir="demo_data/")
    Image("demo_data/prediction_visual.png")"""



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
print(f"geçen zaman {t2 - t1}")