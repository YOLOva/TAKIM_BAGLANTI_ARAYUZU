from cv2 import Mat
import cv2
import numpy as np

from ....utils.image_resize import image_resize


def bbox_to_corners(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]


def sizes_in_range(bbox,min,max):
    sizes=bbox[2:]
    for size in sizes:
        if not (min<=size<=max):
            return False
    return True


def onEdge(bbox):
    error_margin=0.001
    for coordinate in bbox_to_corners(bbox):
        if coordinate<=error_margin or coordinate>=1-error_margin:
            return True
    return False

class CustomFix:
    def __init__(self, frame:Mat) -> None:
        self.frame=frame
    # UAP UAI çok küçük olmamalı +
    # UAP UAI Kenarda ise negatif +
    # UAP UAI Fix With Color +

    # İnsan Boyutu görseldeki kenarda olmayan ve motor dışındakilerden küçük olmalı
    # Bir nesnenin boyutları resmin kenarının yarısından fazla olmamalı


    
    def human_motor_smaller_than_other_classes(self, cocos):
        not_on_edges=list(filter(lambda x: not onEdge(x["bbox"]), cocos))
        """ def get_all_center_objects(cocos):
            return list(filter(lambda x: x["bbox"],cocos)) """
        return cocos
    
    def fix(self, cocos):
        uaip_fixes=UAIPFix(self.frame)
        cocos=uaip_fixes.fix(cocos)
        return cocos
    
    # İnsanların Kutu birleşmesi ya da tek insanın fazladan kutularının olması problemi
    # Deniz taşıtının olacağı yerde mavi alan olması şartı ile yanlış tespit düzeltilebilir.
    #Gölgelerin insana dahil olması

class UAIPFix:
    min_size=0.025
    max_size=0.35
    red_min=0.3
    blue_min=0.3
    def __init__(self, frame:Mat) -> None:
        self.frame=frame
        img = self.frame.copy()
        blurred= cv2.medianBlur(img, 3, img)
        self.blue_mask=self.get_blue_mask(blurred)
        self.red_mask=self.get_red_mask(blurred)
        """ cv2.imshow("test", self.blue_mask)
        cv2.waitKey(0) """

    def get_red_mask(self, frame: Mat) -> Mat:
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # lower mask (0-10)
        lower_red = np.array([0, 255, 60])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([160, 40, 150])
        upper_red = np.array([179, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
        # cv2.imshow("test", mask0+mask1)
        return mask0+mask1
    
    def get_blue_mask(self, frame: Mat) -> Mat:
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 28, 120])
        upper_blue = np.array([110, 140, 255])
        mask1 = cv2.inRange(img_hsv, lower_blue, upper_blue)
        return mask1
    
    def red_blue_ratio(self, bbox):
        h,w=self.frame.shape[:2]
        y1=int(bbox[1]*h)
        y2=int((bbox[1]+bbox[3])*h)
        x1=int(bbox[0]*w)
        x2=int((bbox[0]+bbox[2])*w)
        img=self.frame[y1:y2,x1:x2]
        red_color_ratio=cv2.countNonZero(self.red_mask[y1:y2,x1:x2])/(img.size/3)
        blue_color_ratio=cv2.countNonZero(self.blue_mask[y1:y2,x1:x2])/(img.size/3)
        return red_color_ratio, blue_color_ratio
    
    # Kenardaki positiflerin kontrol edilip düzeltilmesi işlemi
    def get_positives(self, cocos):
        return [coco for coco in cocos if coco["category_id"] in [0,2]]
    def get_uaips(self, cocos):
        return [coco for coco in cocos if coco["category_id"] in [0,1,2,3]]
    def check_on_edge(self, cocos):
        positives = self.get_positives(cocos)
        for positive in positives:
            if onEdge(positive["bbox"]):
                positive["category_id"]+=1
        return cocos
    
    def check_in_size_range(self, cocos):
        not_on_edges=list(filter(lambda x: not onEdge(x["bbox"]), self.get_uaips(cocos)))
        deleteds=[]
        for coco in cocos:
            if not sizes_in_range(coco["bbox"], 0, self.max_size):
                deleteds.append(coco["id"])
        for coco in not_on_edges:
            if not sizes_in_range(coco["bbox"], self.min_size, self.max_size):
                id=coco["id"]
                if id not in deleteds:
                    deleteds.append(coco["id"])
        return list(filter(lambda x: x["id"] not  in deleteds, cocos))
    def check_color(self, cocos):
        deleteds=[]
        for coco in self.get_uaips(cocos):
            red_ratio, blue_ratio = self.red_blue_ratio(coco["bbox"])
            if coco["category_id"] in [0,1] and red_ratio>self.red_min:
                continue
            if coco["category_id"] in [2,3] and blue_ratio>self.blue_min:
                continue
            if red_ratio>self.red_min:
                coco["category_id"]-=2
                continue
            if blue_ratio>self.blue_min:
                coco["category_id"]+=2
                continue
            deleteds.append(coco["id"])
        return list(filter(lambda x: x["id"] not  in deleteds, cocos))

    def fix(self, cocos):
        cocos=self.check_on_edge(cocos)
        cocos=self.check_in_size_range(cocos)
        cocos=self.check_color(cocos)
        return cocos

