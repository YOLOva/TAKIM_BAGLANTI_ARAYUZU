
from src.AutoLabeller.tracker_class import Tracker
import os
from yolov5.utils.plots import Annotator, colors
from PIL import Image
import platform
from numpy import asarray
from pathlib import Path
import argparse
from yolov5.utils.general import (check_requirements, cv2)
import shutil
import statistics

class YoloAnnotation():
    def __init__(self, txt_path) -> None:
        self.txt_path = txt_path
        self.text = ""

    def addObject(self, cls, x_ctr, y_ctr, width, height):
        object_line = f'{cls} {x_ctr} {y_ctr} {width} {height}\n'
        self.text = f"{self.text}{object_line}"

    def write_output(self):
        with open(self.txt_path, 'w') as f:
            f.write(f'{self.text}')
            f.close()


class AutoLabeller:
    def __init__(self, yolo_weights="YOLOva2022Best.pt",
                 label_mapper="label_map_uyz_2023.txt",
                 classes_txt='classes.txt',
                 labels_output_folder="",
                 imgsz=(1088, 1088),
                 device='cuda:0',
                 conf_thres=0.4,
                 show_vid=True,
                 line_thickness=1,
                 check_inilebilir=True) -> None:

        self.tracker = Tracker(yolo_weights=yolo_weights,
                               imgsz=imgsz, device=device, conf_thres=conf_thres)
        self.names = self.tracker.names
        self.labels_output_folder = labels_output_folder
        self.label_mapper = label_mapper
        self.show_vid = show_vid
        self.line_thickness = line_thickness
        self.check_inilebilir = check_inilebilir
        self.classes_txt = classes_txt
        self.detect_index=0

        self.class_map = []
        if label_mapper != "":
            with open(label_mapper, "r") as file:
                lines = file.readlines()
            self.class_map = []
            for line in lines:
                self.class_map.append(int(line.split()[0]))
            print(self.class_map)

        if len(self.class_map) == 0:
            self.class_map = list(range(0, len(self.names)))

    def detect(self, source, save_txt=True, info=None):
        cocos = self.tracker.get_prediction(source)
        if self.detect_index==0 and len(cocos)==0:
            cocos = self.tracker.get_prediction(source)
        cocos = self.fix(cocos)
        if self.show_vid:
            self.draw_box(source, cocos,info)
        if save_txt:
            self.write_labels(cocos, source)
        self.detect_index+=1
        return cocos

    def write_labels(self, cocos, source):
        txt_path = str(Path(self.labels_output_folder))
        Path(txt_path).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.classes_txt, os.path.join(
            txt_path, "classes.txt"))
        yolo_annotation = YoloAnnotation(
            f"{txt_path}\\{Path(source).stem}.txt")

        im = Image.open(source)
        im_w, im_h = im.size
        for coco in cocos:
            bbox = list(coco["bbox"])
            bbox[2] = bbox[0]+bbox[2]
            bbox[3] = bbox[1]+bbox[3]
            cls = coco["category_id"]
            xmin = bbox[0]/im_w
            ymin = bbox[1]/im_h
            xmax = bbox[2]/im_w
            ymax = bbox[3]/im_h
            o_w = xmax-xmin
            o_h = ymax-ymin
            c_x = (xmin+xmax)/2
            c_y = (ymin+ymax)/2
            class_id = self.class_map[cls]
            if self.check_inilebilir and cls in [2, 3] and coco["inilebilir"] == 0:
                class_id = self.class_map[cls]+1
            yolo_annotation.addObject(class_id, c_x, c_y, o_w, o_h)
        yolo_annotation.write_output()

    def draw_box(self, source, cocos, info):
        im0 = Image.open(source)  # .convert('RGB')
        im0 = asarray(im0)
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        annotator = Annotator(
            im0, line_width=self.line_thickness, example=str(self.names))
        for coco in cocos:
            bbox = list(coco["bbox"])
            bbox[2] = bbox[0]+bbox[2]
            bbox[3] = bbox[1]+bbox[3]
            c = int(coco["category_id"])  # integer class
            label = f'{coco["id"]} {self.names[coco["category_id"]]} {coco["score"]:.2f} {coco["inilebilir"]}'
            color = colors(c, True)
            annotator.box_label(bbox, label, color=color)
        # Stream results
        im0 = annotator.result()
        # Choose a font
        im0 = cv2.putText(im0,info, (100, 100), 0, 2, 255,2)
        if self.show_vid:
            p = "video"
            if platform.system() == 'Linux' and p not in self.windows:
                self.windows.append(p)
                # allow window resize (Linux)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL |
                                cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                exit()

    def fix(self, cocos):
        cocos = self.uaips_check(cocos)
        cocos = self.arac_insan_fix(cocos)
        return cocos

    def uaips_check(self, cocos):
        uaips = [coco for coco in cocos if coco["category_id"] in [2, 3]]
        others = [coco for coco in cocos if coco["category_id"] not in [2, 3]]

        for other in others:
            other["inilebilir"] = -1

        for uaip in uaips:
            uaip["inilebilir"] = 1
            uaip_bbox = list(uaip["bbox"])
            uaip_bbox[2] = uaip_bbox[0]+uaip_bbox[2]
            uaip_bbox[3] = uaip_bbox[1]+uaip_bbox[3]
            for other in others:
                other_bbox = list(other["bbox"])
                other_bbox[2] = other_bbox[0]+other_bbox[2]
                other_bbox[3] = other_bbox[1]+other_bbox[3]
                if (  # solüst
                    (uaip_bbox[0] <= other_bbox[0] <= uaip_bbox[2]
                     and uaip_bbox[1] <= other_bbox[1] <= uaip_bbox[3])
                    or  # solalt
                    (uaip_bbox[0] <= other_bbox[0] <= uaip_bbox[2]
                     and uaip_bbox[1] <= other_bbox[3] <= uaip_bbox[3])
                    or  # sağalt
                    (uaip_bbox[0] <= other_bbox[2] <= uaip_bbox[2]
                     and uaip_bbox[1] <= other_bbox[3] <= uaip_bbox[3])
                    or  # sağüst
                    (uaip_bbox[0] <= other_bbox[2] <= uaip_bbox[2]
                     and uaip_bbox[1] <= other_bbox[1] <= uaip_bbox[3])
                ):
                    uaip["inilebilir"] = 0
        return cocos

    def arac_insan_fix(self, cocos):
        fix_ratio = 0.67
        fix_ratio_not_human = 0.8
        araclar = [coco for coco in cocos if coco["category_id"] == 0]
        insanlar = [coco for coco in cocos if coco["category_id"] == 1]
        diger = [coco for coco in cocos if coco["category_id"] not in [0, 1]]
        remove_class_ids = []
        not_human=[]
        if len(araclar)>0 and len(insanlar)>0:
            arac_mw=statistics.median([x["bbox"][2] for x in araclar])
            arac_mh=statistics.median([x["bbox"][3] for x in araclar])
            not_human=[x["id"] for x in insanlar if x["bbox"][2]*fix_ratio_not_human>=arac_mw or x["bbox"][3]*fix_ratio_not_human>=arac_mh ]
        for insan in insanlar:
            id = insan["id"]
            i_w = insan["bbox"][2]
            i_h = insan["bbox"][3]
            insan_bbox = list(insan["bbox"])
            insan_bbox[2] = insan_bbox[0]+insan_bbox[2]
            insan_bbox[3] = insan_bbox[1]+insan_bbox[3]
            try:
                for arac in araclar:
                    a_w = arac["bbox"][2]
                    a_h = arac["bbox"][3]
                    ratiox = min(i_w/a_w, a_w/i_w)
                    ratioy = min(i_h/a_h, a_h/i_h)
                    arac_bbox = list(arac["bbox"])
                    arac_bbox[2] = arac_bbox[0]+arac_bbox[2]
                    arac_bbox[3] = arac_bbox[1]+arac_bbox[3]
                    cx = (arac_bbox[2]+arac_bbox[0])/2
                    cy = (arac_bbox[3]+arac_bbox[1])/2
                    if (insan_bbox[0] <= cx <= insan_bbox[2]
                            and insan_bbox[1] <= cy <= insan_bbox[3]) and ratiox >= fix_ratio and ratioy >= fix_ratio and id not in remove_class_ids:
                        try:
                            remove_class_ids.append(id)
                        except:
                            pass
            except:
                pass
        


        new_insanlar = []
        for insan in insanlar:
            if insan["id"] not in remove_class_ids:
                new_insanlar.append(insan)
            """ if insan["id"] in not_human:
                insan["category_id"]=0
                insan["category_name"]=self.names[0]
                araclar.append(insan) """
        cocos = diger+araclar+new_insanlar
        return cocos



def run(source='',
        yolo_weights='YOLOva2022Best.pt',
        output_folder='',
        conf_thres=0.4,
        label_mapper='label_map_uyz_2023.txt',
        classes_txt='classes.txt',
        check_inilebilir=True,
        device='cuda:0',
        hide_vid=False
        ):

    print(f'source: {source}')
    print(f'output_folder: {output_folder}')

    labeller = AutoLabeller(yolo_weights=yolo_weights, device=device, labels_output_folder=output_folder,
                            show_vid=not hide_vid, conf_thres=conf_thres, check_inilebilir=check_inilebilir, label_mapper=label_mapper, classes_txt=classes_txt)  # "cpu", # or 'cuda:0'
    files = os.listdir(source)
    files = [os.path.join(source, f) for f in files] # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))
    for file in files:
        cocos=labeller.detect(source=file)


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='', help='dir')
    parser.add_argument('--output-folder', type=str, default='', help='dir')
    parser.add_argument('--yolo-weights', type=str,
                        default="YOLOva2022Best.pt", help='model.pt path')
    parser.add_argument('--classes-txt', type=str,
                        default="classes.txt", help='classes.txt path')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='confidence threshold')
    parser.add_argument('--label-mapper', type=str,
                        default='label_map_uyz_2023.txt', help='labelmapper path')
    parser.add_argument('--check-inilebilir', action='store_true',
                        help='inilebilirlik durumunu tespit et')
    parser.add_argument('--device', default='cuda:0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--hide-vid', action='store_true',
                        help='display tracking video results')
    opt = parser.parse_args()
    # opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt',
                       exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
