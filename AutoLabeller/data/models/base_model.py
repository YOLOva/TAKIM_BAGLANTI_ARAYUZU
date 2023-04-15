
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, POSTPROCESS_NAME_TO_CLASS



self.model = AutoDetectionModel.from_pretrained(
            model_type='yolov5',
            model_path=params.model,
            model=""
            confidence_threshold=params.conf,
            image_size=params.imgsz,
            device=params.device  # "cpu", # or 'cuda:0'
        )