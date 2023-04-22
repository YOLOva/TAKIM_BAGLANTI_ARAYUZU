from cv2 import Mat
import cv2
import numpy as np

from ....utils.image_resize import image_resize


# frame_devrim max:0.552778, others 0.31247174298321767
uaip_size_range = [0.025, 0.35]
human_range = [0.005208, 0.2]  # 0.004167, 0.08796296296296291
moto_range = [0.01, 0.2]  # 0.010417,  0.098148
others_range = [0.01, 0.31]  # otomobil 0.012037, 0.305556
# kamyon diğer kara:0.014583, 0.498148
truck_train_ship_range = [others_range[0], 0.63]

size_range_map = {0: uaip_size_range,
                  1: uaip_size_range,
                  2: uaip_size_range,
                  3: uaip_size_range,
                  4: human_range,
                  5: others_range,
                  6: moto_range,
                  7: truck_train_ship_range,
                  8: truck_train_ship_range,
                  9: truck_train_ship_range,
                  10: truck_train_ship_range,
                  11: truck_train_ship_range}

others_conf=0.6#0.52
car_conf=0.6#0.54
uaips_conf=0.6
human_conf=0.48
small_object_conf=0.44
conf_range_map = {0: {"center":uaips_conf, "edge":uaips_conf},
                  1: {"center":uaips_conf, "edge":uaips_conf},
                  2: {"center":uaips_conf, "edge":uaips_conf},
                  3: {"center":uaips_conf, "edge":uaips_conf},
                  4: {"center":human_conf, "edge":small_object_conf},
                  5: {"center":car_conf, "edge":0.55},
                  6: {"center":small_object_conf, "edge":small_object_conf},
                  7: {"center":others_conf, "edge":small_object_conf},
                  8: {"center":others_conf, "edge":small_object_conf},
                  9: {"center":others_conf, "edge":small_object_conf},
                  10: {"center":others_conf, "edge":small_object_conf},
                  11: {"center":others_conf, "edge":small_object_conf}}


def average(numbers):
    return sum(numbers) / len(numbers)


def bbox_to_corners(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]


def sizes_in_range(bbox, range):
    sizes = bbox[2:]
    for size in sizes:
        if not (range[0] <= size <= range[1]):
            return False
    return True


def onEdge(bbox, error_margin=0.002):
    for coordinate in bbox_to_corners(bbox):
        if coordinate <= error_margin or coordinate >= 1-error_margin:
            return True
    return False

def filter_with_id_list(cocos, deleteds):
    return list(filter(lambda x: x["id"] not in deleteds, cocos))

def get_cocos_of_id(ids, cocos):
    return list(filter(lambda x: x["category_id"] in ids, cocos))
class CustomFix:
    def __init__(self, frame: Mat) -> None:
        self.frame = frame
    # UAP UAI çok küçük olmamalı +
    # UAP UAI Kenarda ise negatif +
    # UAP UAI Fix With Color +

    # İnsan Boyutu görseldeki kenarda olmayan ve motor dışındakilerden küçük olmalı+
    # Bir nesnenin boyutları resmin kenarının yarısından fazla olmamalı
    # Boyut limitleri belirlenmeli, bunları aşanlar elenmeli +
    def min_confidence_fix(self, cocos):
        deleteds = []
        for coco in cocos:
            cls = coco["category_id"]
            conf_range = conf_range_map[cls]
            if not onEdge(coco["bbox"]) and coco["score"]<=conf_range["center"]:
                deleteds.append(coco["id"])
                continue
            if coco["score"]<=conf_range["edge"]:
                deleteds.append(coco["id"])

        return filter_with_id_list(cocos, deleteds)


    def size_fix(self, cocos):
        deleteds = []
        for coco in cocos:
            cls = coco["category_id"]
            size_range = size_range_map[cls]
            bbox = coco["bbox"]
            if not onEdge(bbox) and not sizes_in_range(bbox, size_range):
                deleteds.append(coco["id"])
                continue
            if not sizes_in_range(bbox, [0, size_range[1]]):
                deleteds.append(coco["id"])
        return filter_with_id_list(cocos, deleteds)

    def fix(self, cocos):
        cocos=self.min_confidence_fix(cocos)
        cocos = self.size_fix(cocos)
        uaip_fixes = UAIPFix(self.frame)
        cocos = uaip_fixes.fix(cocos)
        human_fixes = HumanAndMotoFix(self.frame)
        cocos = human_fixes.fix(cocos)
        return cocos

    # İnsanların Kutu birleşmesi ya da tek insanın fazladan kutularının olması problemi
    # Deniz taşıtının olacağı yerde mavi alan olması şartı ile yanlış tespit düzeltilebilir.
    # Gölgelerin insana dahil olması


class HumanAndMotoFix:
    def __init__(self, frame:Mat) -> None:
        self.frame=frame
        self.green_mask=self.get_green_mask(frame.copy())
        """ cv2.imshow("green", self.green_mask)
        cv2.imshow("frame", frame)
        cv2.waitKey(0) """
    def get_green_mask(self, frame: Mat) -> Mat:
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([30, 100, 0])
        upper_blue = np.array([60, 255, 255])
        mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
        return mask
    
    def compare_size_fix(self, cocos):
        not_on_edges = list(filter(lambda x: not onEdge(
            x["bbox"]) and x["category_id"] not in [4, 6], cocos))
        human_and_motors = list(
            filter(lambda x: x["category_id"] in [4, 6], cocos))
        ws = [x for x in not_on_edges["bbox"][2]]
        hs = [x for x in not_on_edges["bbox"][3]]
        min_w, min_h = 1, 1
        if not_on_edges:
            min_w = min(ws)
            min_h = min(hs)
        deleteds = []
        for coco in human_and_motors:
            bbox = coco["bbox"]
            if len(not_on_edges) > 0 and (bbox[2] > min_w or bbox[3] > min_h or bbox[2] < min_w*0.25 or bbox[3] < min_h*0.25):
                deleteds.append(coco["id"])
                continue
        return filter_with_id_list(cocos, deleteds)
    
    def green_ratio(self, bbox):
        h, w = self.frame.shape[:2]
        y1 = int(bbox[1]*h)
        y2 = int((bbox[1]+bbox[3])*h)
        x1 = int(bbox[0]*w)
        x2 = int((bbox[0]+bbox[2])*w)
        img = self.frame[y1:y2, x1:x2]
        color_ratio = cv2.countNonZero(
            self.green_mask[y1:y2, x1:x2])/(img.size/3)
        return color_ratio
    
    def green_area_filter(self, cocos):
        humans=get_cocos_of_id([4], cocos)
        deleteds=[]
        for coco in humans:
            color_ratio=self.green_ratio(coco["bbox"])
            #print(color_ratio)
            if color_ratio>0.88:
                deleteds.append(coco["id"])
            else:
                print(color_ratio)
                """ cv2.imshow("green", self.green_mask)
                cv2.waitKey(0) """
        return filter_with_id_list(cocos, deleteds)

    def fix(self, cocos):
        # cocos = self.compare_size_fix(cocos)
        cocos=self.green_area_filter(cocos)
        return cocos


class UAIPFix:
    red_min = 0.3
    blue_min = 0.3

    def __init__(self, frame: Mat) -> None:
        self.frame = frame
        img = self.frame.copy()
        blurred = cv2.medianBlur(img, 3, img)
        self.blue_mask = self.get_blue_mask(blurred)
        self.red_mask = self.get_red_mask(blurred)

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
        h, w = self.frame.shape[:2]
        y1 = int(bbox[1]*h)
        y2 = int((bbox[1]+bbox[3])*h)
        x1 = int(bbox[0]*w)
        x2 = int((bbox[0]+bbox[2])*w)
        img = self.frame[y1:y2, x1:x2]
        red_color_ratio = cv2.countNonZero(
            self.red_mask[y1:y2, x1:x2])/(img.size/3)
        blue_color_ratio = cv2.countNonZero(
            self.blue_mask[y1:y2, x1:x2])/(img.size/3)
        return red_color_ratio, blue_color_ratio

    # Kenardaki positiflerin kontrol edilip düzeltilmesi işlemi
    def get_positives(self, cocos):
        return [coco for coco in cocos if coco["category_id"] in [0, 2]]

    def get_uaips(self, cocos):
        return [coco for coco in cocos if coco["category_id"] in [0, 1, 2, 3]]

    def check_on_edge(self, cocos):
        positives = self.get_positives(cocos)
        for positive in positives:
            if onEdge(positive["bbox"], error_margin=0.001):
                positive["category_id"] += 1
        return cocos

    def check_color(self, cocos):
        deleteds = []
        for coco in self.get_uaips(cocos):
            red_ratio, blue_ratio = self.red_blue_ratio(coco["bbox"])
            if coco["category_id"] in [0, 1] and red_ratio > self.red_min:
                continue
            if coco["category_id"] in [2, 3] and blue_ratio > self.blue_min:
                continue
            if red_ratio > self.red_min:
                coco["category_id"] -= 2
                continue
            if blue_ratio > self.blue_min:
                coco["category_id"] += 2
                continue
            deleteds.append(coco["id"])
        return filter_with_id_list(cocos, deleteds)

    def fix(self, cocos):
        cocos = self.check_on_edge(cocos)
        cocos = self.check_color(cocos)
        return cocos
