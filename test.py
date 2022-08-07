import logging
from datetime import datetime
from pathlib import Path

from decouple import config

from src.connection_handler import ConnectionHandler
from src.frame_predictions import FramePredictions
from src.object_detection_model import ObjectDetectionModel
from src.yolova.status_saver import PredictStatusSaver

def configure_logger(team_name):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(log_folder + team_name + '_%Y_%m_%d__%H_%M_%S_%f.log')
    """for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)"""
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def run():
    print("Started...")
    config.search_path = "./config/"
    team_name = config('TEAM_NAME')
    password = config('PASSWORD')
    evaluation_server_url = config("EVALUATION_SERVER_URL")
    configure_logger(team_name)
    detection_model = ObjectDetectionModel(evaluation_server_url)
    server = ConnectionHandler(evaluation_server_url, username=team_name, password=password)
    frames_json = server.get_frames()
    images_folder = "./_images/"
    Path(images_folder).mkdir(parents=True, exist_ok=True)
    start_index = 0
    for index, frame in enumerate(frames_json[start_index:], start=start_index):
        print(f"current index {index}")
        predictions = FramePredictions(frame['url'], frame['image_url'], frame['video_name'])
        predictions = detection_model.process(index, predictions,evaluation_server_url)
if __name__ == '__main__':
    run()
