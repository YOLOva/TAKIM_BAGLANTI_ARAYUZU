import math
import cv2
import numpy as np
from ...labeller.fix.fix import AllClassesFixs

from ...utils.image_resize import image_resize


class UAIPFix:
    frame=None
    all_classes_fix:AllClassesFixs
    
    edge_margin=0.0015
    ratio_margin=0.5
    
    def __init__(self, frame, all_classes_fix) -> None:
        self.frame=frame
        self.all_classes_fix=all_classes_fix

    def fix(self, cocos):
        cocos=self.different_class_in_class_fix(cocos)
        cocos=self.fix_with_colors(cocos)
        cocos=self.detect_is_on_edge_and_square_dimension(cocos)
        cocos=self.detect_area_is_empty(cocos)
        cocos=self.check_and_fix_not_square_and_not_in_edge(cocos)
        cocos=self.detect_uaip_in_uaip(cocos)
        return cocos

    def redMask(self, img_hsv):
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([160, 40, 60])
        upper_red = np.array([179, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
        mask = mask0+mask1
        return mask
    
    def blueMask(self, img_hsv):
        lower_blue = np.array([90,28,200],np.uint8)
        upper_blue = np.array([110,255,255],np.uint8)
        mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
        return mask
    def isThereCircle(self, bbox, red=True):
        img = self.frame.copy()
        h,w=img.shape[:2]

        y1=int(bbox[1]*h)
        y2=int((bbox[1]+bbox[3])*h)
        x1=int(bbox[0]*w)
        x2=int((bbox[0]+bbox[2])*w)

        img = img[y1:y2,x1:x2]
        img = cv2.medianBlur(img, 3, img)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask =  self.redMask(img_hsv) if red else self.blueMask(img_hsv)

        output_img = img.copy()
        output_img[np.where(mask == 0)] = 0

        # or your HSV image, which I *believe* is what you want
        output_hsv = img_hsv.copy()
        output_hsv[np.where(mask == 0)] = 0

        gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 80,
                                param1=gray.shape[0]/4,
                                param2=30,
                                minRadius=0,
                                maxRadius=0)
        if circles is not None:
            #circles = np.uint16(np.around(circles))
            return True
        return False
    def isRed(self, bbox, red=True):
        img = self.frame.copy()
        img=image_resize(img, img.shape[1]*2)
        h,w=img.shape[:2]

        y1=int(bbox[1]*h)
        y2=int((bbox[1]+bbox[3])*h)
        x1=int(bbox[0]*w)
        x2=int((bbox[0]+bbox[2])*w)

        img = img[y1:y2,x1:x2]
        img = cv2.medianBlur(img, 3, img)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask =  self.redMask(img_hsv) if red else self.blueMask(img_hsv)
        color_ratio=cv2.countNonZero(mask)/(img.size/3)
        
        output_img = img.copy()
        output_img[np.where(mask == 0)] = 0
        """ cv2.imshow("test", img)
        cv2.waitKey(0) """
        return color_ratio>=0.3, color_ratio
    def different_class_in_class_fix(self, cocos):
        uaips = [coco for coco in cocos if coco["category_id"] in [2, 3]]
        remove_ids=[]
        # İç içe iniş alanı varsa Renk kontrolü ile düzeltilir.
        for uaip in uaips:
            if uaip["category_id"] in remove_ids:
                continue
            for uaip2 in uaips:
                if uaip==uaip2:
                    continue
                if self.all_classes_fix.coco_in_and_same_size_coco(uaip2, uaip, False):
                    bbox=uaip["bbox"] if uaip["bbox"][2]>=uaip2["bbox"][2] and uaip["bbox"][3]>=uaip2["bbox"][3] else uaip2["bbox"]
                    areaIsRed,r=self.isRed(bbox)
                    if areaIsRed and uaip2["category_id"] == 2 or not areaIsRed and uaip2["category_id"] == 3:
                        remove_ids.append(uaip2["id"])
        return list(filter(lambda x: x["id"] not in remove_ids, cocos))
    
    def fix_with_colors(self, cocos):
        uaips = [coco for coco in cocos if coco["category_id"] in [2, 3]]
        remove_ids=[]
        for uaip in uaips:
            if uaip["category_id"] == 2:
                isBlue,r=self.isRed(uaip["bbox"], red=False)
                if not isBlue:
                    isRed, r=self.isRed(uaip["bbox"])
                    if isRed:
                        uaip["category_id"]=3
                    else:
                        remove_ids.append(uaip["id"])
            
            if uaip["category_id"] == 3:
                isRed,r=self.isRed(uaip["bbox"])
                if not isRed:
                    isBlue,r=self.isRed(uaip["bbox"], red=False)
                    if isBlue:
                        uaip["category_id"]=2
                    else:
                        remove_ids.append(uaip["id"])
        return list(filter(lambda x: x["id"] not in remove_ids, cocos))
    
    def in_rectangle(self, rect_bbox, bbox):
        centerx=rect_bbox[0]+rect_bbox[2]/2
        centery=rect_bbox[1]+rect_bbox[3]/2
        ax=bbox[0]
        ay=bbox[1]
        bx=bbox[0]+bbox[2]
        by=bbox[1]+bbox[3]
        corners=[(ax, ay), (ax, by), (bx, ay), (bx, by)]
        for corner in corners:
            x=corner[0]
            y=corner[1]

            if x>=centerx-rect_bbox[2] and x<=centerx+rect_bbox[2] and y>=centery-rect_bbox[3]and y<=centery+rect_bbox[3]:
                return True
        return False

    def in_circle(self,circle_bbox, bbox):
        c_bbox=list([x*self.frame.shape[0] if i%2==1 else x*self.frame.shape[1] for i, x in enumerate(circle_bbox)])
        nbbox=list([x*self.frame.shape[0] if i%2==1 else x*self.frame.shape[1] for i, x in enumerate(bbox)])
        center_x=(c_bbox[0]+c_bbox[2]/2)
        center_y=(c_bbox[1]+c_bbox[3]/2)
        radius=max(c_bbox[2], c_bbox[3])/2
        radius2=radius**2
        radiusa=max(c_bbox[2], c_bbox[3])/2
        ax=nbbox[0]
        ay=nbbox[1]
        bx=(nbbox[0]+nbbox[2])
        by=(nbbox[1]+nbbox[3])
        corners=[(ax, ay), (ax, by), (bx, ay), (bx, by)]
        """ cv2.circle(self.frame, (int(center_x), int(center_y)), int(radius),(0,255,0), 2) """
        for corner in corners:
            x=corner[0]
            y=corner[1]
            if x>=center_x-c_bbox[2] and x<=center_x+c_bbox[2] and y>=center_y-c_bbox[3]and y<=center_y+c_bbox[3]:
                r2=(x-center_x)**2+(y-center_y)**2
                if r2<=radius2:
                    return True
        return False
    def in_ellipse(self, ellips_bbox, bbox):
        c_bbox=list([x*self.frame.shape[0] if i%2==1 else x*self.frame.shape[1] for i, x in enumerate(ellips_bbox)])
        nbbox=list([x*self.frame.shape[0] if i%2==1 else x*self.frame.shape[1] for i, x in enumerate(bbox)])
        h=(c_bbox[0]+c_bbox[2]/2)
        k=(c_bbox[1]+c_bbox[3]/2)
        a=c_bbox[2]/2
        b=c_bbox[3]/2
        def checkpoint(h, k, x, y, a, b):
            # checking the equation of
            # ellipse with the given point
            p = ((math.pow((x - h), 2) / math.pow(a, 2)) +
                (math.pow((y - k), 2) / math.pow(b, 2)))
        
            return p
        
        ax=nbbox[0]
        ay=nbbox[1]
        bx=(nbbox[0]+nbbox[2])
        by=(nbbox[1]+nbbox[3])
        corners=[(ax, ay), (ax, by), (bx, ay), (bx, by)]
        for corner in corners:
            x=corner[0]
            y=corner[1]
            if checkpoint(h, k, x, y, a, b) <= 1:
                return True
        return False
    




    def bbox_in_bbox(self, uaip_bbox, other_bbox):
        ax = uaip_bbox[0]+uaip_bbox[2]
        ay = uaip_bbox[1]+uaip_bbox[3]
        bx = other_bbox[0]+other_bbox[2]
        by = other_bbox[1]+other_bbox[3]

        oax=other_bbox[0]
        oay=other_bbox[1]
        obx=other_bbox[3]+oax
        oby=other_bbox[3]+oay
        return (  # solüst
                    (ax <= oax <= bx
                     and ay <= oay <= by)
                    or  # solalt
                    (ax <= oax <= bx
                     and ay <= oby <= by)
                    or  # sağalt
                    (ax <= obx <= bx
                     and ay <= oby <= by)
                    or  # sağüst
                    (ax <= obx <= bx
                     and ay <= oay <= by)
                )

    def detect_area_is_empty(self, cocos):
        uaips = [coco for coco in cocos if coco["category_id"] in [2, 3]]
        
        others = [coco for coco in cocos if coco["category_id"] not in [2, 3]]
        for uaip in filter(lambda x: x["inilebilir"]==1, uaips):
            for other in others:
                if self.in_ellipse(uaip["bbox"], other["bbox"]):
                    uaip["inilebilir"] = 0
                #uaip["inilebilir"] = 0 if self.bbox_in_bbox(uaip, other) else 1

        return cocos
    def detect_is_on_edge_and_square_dimension(self, cocos):
        uaips = [coco for coco in cocos if coco["category_id"] in [2, 3]]
        for uaip in uaips:
            # köşelerde ise
            bbox=list(uaip["bbox"])
            # kenarlarda ise
            cornerx=bbox[0]+bbox[2]
            cornery=bbox[1]+bbox[3]
            
            uaip["in_edge"]=False
            uaip["not_square"]=False
            if bbox[0]<=self.edge_margin or bbox[1]<=self.edge_margin or cornerx>=1-self.edge_margin or cornery>=1-self.edge_margin:
                uaip["inilebilir"] = 0
                uaip["in_edge"]=True
            ratio=min(bbox[2]/bbox[3], bbox[3]/bbox[2])

            if ratio<=1-self.ratio_margin:
                uaip["inilebilir"] = 0
                uaip["not_square"]=True
            
        return cocos
    def get_all_area_of_color(self, bbox, red=True):
            img = self.frame.copy()
            img=image_resize(img, img.shape[1]*2)
            h,w=img.shape[:2]
            
            bbox=list([x*img.shape[0] if i%2==1 else x*img.shape[1] for i, x in enumerate(bbox)])
            diff=int(abs(bbox[2]-bbox[3]))
            y1=int(bbox[1])
            y2=int((bbox[1]+bbox[3]))
            x1=int(bbox[0])
            x2=int((bbox[0]+bbox[2]))
            y1=y1-diff
            if y1<0: y1=0
            y2=y2+diff
            if y2>h: y2=h
            x1=x1-diff
            if x1<0: x1=0
            x2=x2+diff
            if x2>w: x2=w

            img = img[y1:y2,x1:x2]
            img = cv2.medianBlur(img, 3, img)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask =  self.redMask(img_hsv) if red else self.blueMask(img_hsv)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt=max(contours, key = cv2.contourArea)
            x,y,w,h=cv2.boundingRect(cnt)
            return [x1+x, y1+y, w, h]


            """ rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            xs=[x[0] for x in box]
            ys=[x[1] for x in box]
            minx=min(xs)/w
            miny=min(ys)/h
            maxx=max(xs)/w
            maxy=max(ys)/h
            return [x1+minx, y1+miny, maxx-minx, maxy-miny] """

    def check_and_fix_not_square_and_not_in_edge(self, cocos):
        uaips = [coco for coco in cocos if coco["category_id"] in [2, 3] and coco["not_square"] and not coco["in_edge"]]
        for uaip in uaips:
            uaip["bbox"]=self.get_all_area_of_color(uaip["bbox"], uaip["category_id"]==3)
            bbox=uaip["bbox"]
            ratio=min(bbox[2]/bbox[3], bbox[3]/bbox[2])
            uaip["in_edge"]=False
            cornerx=bbox[0]+bbox[2]
            cornery=bbox[1]+bbox[3]
            if bbox[0]<=self.edge_margin or bbox[1]<=self.edge_margin or cornerx>=1-self.edge_margin or cornery>=1-self.edge_margin:
                uaip["inilebilir"] = 0
                uaip["in_edge"]=True
            ratio=min(bbox[2]/bbox[3], bbox[3]/bbox[2])

            uaip["not_square"]=False
            if uaip["inilebilir"] !=0 and ratio<=1-self.ratio_margin:
                uaip["inilebilir"] = 0
                uaip["not_square"]=True
        return cocos

    def detect_uaip_in_uaip(self, cocos):
        uaips = [coco for coco in cocos if coco["category_id"] in [2, 3]]        
        nremoves=[x["id"] for x in self.all_classes_fix.multiple_box_in_box_fix(uaips)]
        removes=[x["id"] for x in uaips if x["id"] not in nremoves]
        return list(filter(lambda x: x["id"] not in removes, cocos))
    