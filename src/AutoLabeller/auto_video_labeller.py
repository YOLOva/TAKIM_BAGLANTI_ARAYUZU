
import argparse
import os
from pathlib import Path
import shutil
from src.AutoLabeller.auto_labeller import AutoLabeller
from yolov5.utils.general import (check_requirements, cv2)


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized




def run(source='',
        yolo_weights='YOLOva2022Best.pt',
        output_folder='',
        conf_thres=0.4,
        label_mapper='label_map_uyz_2023.txt',
        classes_txt='classes.txt',
        check_inilebilir=True,
        device='cuda:0',
        hide_vid=False,
        frame_per_second_to_save=1,
        resize_image = True
        ):

    print(f'source: {source}')
    print(f'output_folder: {output_folder}')

    
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if frame_per_second_to_save <= 0:
        frame_per_second_to_save = 1
    
    vidcap = cv2.VideoCapture(source)
    if int(major_ver) < 3:
            fps = round(vidcap.get(cv2.cv.CV_CAP_PROP_FPS))
            print(
                "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = round(vidcap.get(cv2.CAP_PROP_FPS))
        print(
            "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( length )
    video_name = Path(source).stem  # video ismi alınır kaydetmek için
    count = 0  # sayıcı
    # resimleri kaydetme klasörü
    image_save_folder = os.path.join(output_folder, video_name, "images")
    label_save_folder = os.path.join(output_folder, video_name, "labels")
    # kaydedilecek klasör yoksa oluşturur.
    Path(image_save_folder).mkdir(exist_ok=True, parents=True)
    Path(label_save_folder).mkdir(exist_ok=True, parents=True)

    t_image_per_second = frame_per_second_to_save if fps > frame_per_second_to_save else fps
    success=True
    dt_frame="detect_frame.jpg"
    
    labeller = AutoLabeller(yolo_weights=yolo_weights, device=device, labels_output_folder=label_save_folder,
                            show_vid=not hide_vid, conf_thres=conf_thres, check_inilebilir=check_inilebilir, label_mapper=label_mapper, classes_txt=classes_txt)  # "cpu", # or 'cuda:0'
    while success:  # resimleri bitene kadar devam eder
        save= count % round(fps/t_image_per_second) == 0
        success, image = vidcap.read()
        if not success:
            continue
        save_image_path=os.path.join(image_save_folder, f"frame_{str(count).zfill(6)}.jpg")
        (h,w) = image.shape[:2]
        if resize_image and h>1080:
            image = image_resize(image, height = 1080)
        cv2.imwrite(dt_frame, image)
        if save:
            shutil.copyfile(dt_frame, save_image_path)
        source = save_image_path if save else dt_frame
        cocos=labeller.detect(source=source, save_txt=save, info=f"{count}/{length}")
        count += 1



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='', help='dir')
    parser.add_argument('--output-folder', type=str, default='', help='dir')
    parser.add_argument('--frame-per-second-to-save', type=int,
                        default=1, help='frame_per_second_to_save')
    parser.add_argument('--resize-image', type=int,
                        default=True, help='frame_per_second_to_save')
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