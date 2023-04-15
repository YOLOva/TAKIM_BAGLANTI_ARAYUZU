import cv2
class KCF_Tracker:
    def __init__(self) -> None:
        self.tracker = cv2.TrackerKCF_create()
    
#https://learnopencv.com/object-tracking-using-opencv-cpp-python/