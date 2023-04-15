import cv2
class BasePredictor:
    frame:cv2.Mat=None
    def get_prediction(self, source: str):
        pass