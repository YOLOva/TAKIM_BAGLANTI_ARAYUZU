import json
from pathlib import Path
import sys
import time

from tkinter import Tk, Canvas, PhotoImage, messagebox
from AutoLabeller.utils import image_resize
from src.connection_handler import ConnectionHandler
from src.frame_predictions import FramePredictions
from src.object_detection_model import ObjectDetectionModel

from widgets import CanvasButton
from pathlib import Path
import cv2
from PIL import ImageTk, Image
from numpy import asarray
from yolov5.utils.plots import Annotator, colors
from src.yolova.status_saver import PredictStatusSaver, SendedPrediction
from threading import Thread
import logging
from datetime import datetime
from decouple import config


ASSETS_PATH = "./assets"


def relative_to_assets(path: str) -> Path:
    return Path(__file__).parent / ASSETS_PATH / Path(path)


def configure_logger(team_name):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(
        log_folder + team_name + '_%Y_%m_%d__%H_%M_%S_%f.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


class TeamGUI:
    after = None

    def __init__(self) -> None:
        self.window = Tk()
        self.window.geometry("1920x1080")
        # setting attribute
        self.window.attributes('-fullscreen', True)
        self.canvas = Canvas(
            self.window,
            bg="#000000",
            height=1080,
            width=1920,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)
        image_image_1 = PhotoImage(
            file=relative_to_assets("background.png"))
        self.current_image = self.canvas.create_image(
            960.0,
            540.0,
            image=image_image_1
        )
        self.startbtn = CanvasButton(
            self.canvas, 42, 38, relative_to_assets("start.png"), self.start)
        self.clearbtn = CanvasButton(
            self.canvas, 254, 38, relative_to_assets("clear.png"), self.clear)
        exit_appbtn = CanvasButton(
            self.canvas, 1824.0, 36.0, relative_to_assets("exit_app.png"), self.destroy)
        minimizebtn = CanvasButton(
            self.canvas, 1745.0, 36.0, relative_to_assets("minimize.png"), self.minimize)

        image_image_2 = PhotoImage(
            file=relative_to_assets("info_back2.png"))
        image_2 = self.canvas.create_image(
            222.0,
            908.0,
            image=image_image_2
        )
        
        logobtn=CanvasButton(self.canvas, 1730.0, 860.0, relative_to_assets("logo.png"), self.logoClick)
        self.infolabel = self.canvas.create_text(
            16.0,
            809.0,
            anchor="nw",
            text="Başarılar :)",
            fill="#FFFFFF",
            font=("Inter Black", 24 * -1)
        )
        self.canvas.itemconfig(self.infolabel)
        self.percentlabel = self.canvas.create_text(
            292.0,
            780.0,
            anchor="nw",
            text="0/0",
            fill="#FFFFFF",
            font=("Inter Black", 24 * -1)
        )
        self.status_saver = PredictStatusSaver()
        if len(self.status_saver.sended_predictions) == 0:
            self.clearbtn.show(False)
        self.window.resizable(False, False)
        self.window.mainloop()
    def logoClick(self):
        messagebox.showinfo("Bilgi","Emre Aydemir Gururla Sunar.")
    def destroy(self):
        if self.after is not None:
            self.window.after_cancel(self.after)
        sys.exit()

    def minimize(self):
        self.window.iconify()

    def start(self):
        self.process_thread()

    def clear(self):
        self.status_saver.clear()
        if len(self.status_saver.sended_predictions) == 0:
            self.clearbtn.show(False)

    def update_text(self, message, predictions: FramePredictions, success_count, fail_count, index, max_index):
        process_text = f"""
        {message}
        Başarılı Gönderim Sayısı: {success_count}
        Başarısız Gönderim Sayısı: {fail_count}
        İndirmede geçen süre: {round(predictions.download_time,2)}s
        Tespit süresi:{round(predictions.detection_time,2)}s
        Gönderimde geçen süre: {round(predictions.send_time,2)}s
        80 frame bekleme süresi:{round(predictions.sleep_time,2)}s
        Geçen süre:{round(predictions.total_time,2)}s"""
        percent_text = f"{index}/{max_index}"
        self.canvas.itemconfig(self.infolabel, text=process_text)
        self.canvas.itemconfig(self.percentlabel, text=percent_text)

    def update_text2(self, process_text, percent_text="0/0"):
        self.canvas.itemconfig(self.infolabel, text=process_text)
        self.canvas.itemconfig(self.percentlabel, text=percent_text)

    def change_image(self, frame):
        img = Image.fromarray(frame)
        self.current_image_image = ImageTk.PhotoImage(image=img)
        self.canvas.itemconfig(
            self.current_image, image=self.current_image_image)

    def draw_box(self, prediction: FramePredictions):
        frame = image_resize(cv2.imread(prediction.image_path))

        im0 = asarray(frame)
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        annotator = Annotator(
            im0, line_width=1, example=str(prediction.names))
        for coco in prediction.cocos:
            bbox = list(coco["bbox"])
            bbox[2] = bbox[0]+bbox[2]
            bbox[3] = bbox[1]+bbox[3]
            c = int(coco["category_id"])  # integer class
            label = f'{coco["id"]} {coco["category_name"]} {coco["score"]:.2f} {coco["inilebilir"]}'
            color = colors(c, False)
            annotator.box_label(bbox, label, color=color)
        # Stream results
        im0 = annotator.result()
        self.after = self.window.after(50, self.change_image, im0)

    def process_thread(self):
        try:
            self.main_thread = Thread(
                target=self.start_process, daemon=True).start()
        except (KeyboardInterrupt, SystemExit):
            self.destroy()

    def start_process(self):
        self.startbtn.show(False)
        self.clearbtn.show(False)
        config.search_path = "./config/"
        team_name = config('TEAM_NAME')
        password = config('PASSWORD')
        evaluation_server_url = config("EVALUATION_SERVER_URL")
        configure_logger(team_name)
        detection_model = ObjectDetectionModel(evaluation_server_url)
        server = ConnectionHandler(
            evaluation_server_url, username=team_name, password=password)
        frames_json = server.get_frames()
        images_folder = "./_images/"
        Path(images_folder).mkdir(parents=True, exist_ok=True)
        num_error = 0
        num_success = 0
        if len(frames_json) == 0:
            self.update_text2("Frame listesi boş!")
        else:
            
            for index, frame in enumerate(frames_json):
                if (not self.status_saver.isSended(index)):
                    t1_total = time.perf_counter()
                    predictions = FramePredictions(
                        frame['url'], frame['image_url'], frame['video_name'], frame["session"].split("/")[-2])
                    predictions = detection_model.process(
                        index, predictions, evaluation_server_url)
                    self.draw_box(prediction=predictions)
                    tsend = time.perf_counter()

                    # 80 Frame Limiti Çözümü
                    if len(self.status_saver.sended_predictions) >= 80:
                        before_80 = self.status_saver.sended_predictions[-80]
                        passed_between_80 = tsend - before_80.sendTime
                        if passed_between_80 <= 60:
                            predictions.sleep_time=60-passed_between_80
                            time.sleep(predictions.sleep_time)

                    result = server.send_prediction(predictions)
                    t2_total = time.perf_counter()
                    predictions.total_time = t2_total-t1_total
                    predictions.send_time = t2_total-tsend
                    response_json = json.loads(result.text)
                    success = True
                    message = "Gönderim Başarılı"
                    if result.status_code == 201:
                        self.status_saver.addSendedPrediction(
                            SendedPrediction(index, t2_total))

                    elif "You have already send prediction for this frame." in response_json["detail"]:
                        self.status_saver.addSendedPrediction(
                            SendedPrediction(index, t2_total))
                        message = "Frame daha önce gönderilmiş"
                    # dakikada 80 limiti aşılmışsa
                    elif ("You do not have permission to perform this action." in response_json["detail"] 
                    or "Your requests has been exceeded 80/m limit." in response_json["detail"]):
                        num_error += 1
                        success = False
                        message = "Dakika'da 80 frame limiti aşıldı"
                    else:
                        num_error += 1
                        success = False
                        message = response_json["detail"]
                    if success:
                        num_success += 1
                    self.update_text(message=message, predictions=predictions, success_count=num_success,
                                     fail_count=num_error, index=index+1, max_index=len(frames_json))
                else:
                    num_success += 1
            self.update_text2(
                "Oturum Tamamlandı!", f"{num_success}/{len(frames_json)}")
        self.startbtn.show(True)
        if len(self.status_saver.sended_predictions) > 0:
            self.clearbtn.show(True)
