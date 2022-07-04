import concurrent.futures
import logging
from datetime import datetime
from pathlib import Path
import time
import json

from decouple import config

from src.connection_handler import ConnectionHandler
from src.frame_predictions import FramePredictions
from src.object_detection_model import ObjectDetectionModel


def configure_logger(team_name):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(log_folder + team_name + '_%Y_%m_%d__%H_%M_%S_%f.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def run():
    print("Started...")
    # Get configurations from .env file
    config.search_path = "./config/"
    team_name = config('TEAM_NAME')
    password = config('PASSWORD')
    evaluation_server_url = config("EVALUATION_SERVER_URL")

    # Declare logging configuration.
    configure_logger(team_name)

    # Teams can implement their codes within ObjectDetectionModel class. (OPTIONAL)
    detection_model = ObjectDetectionModel(evaluation_server_url)

    # Connect to the evaluation server.
    server = ConnectionHandler(evaluation_server_url, username=team_name, password=password)

    # Get all frames from current active session.
    frames_json = server.get_frames()

    # Create images folder
    images_folder = "./_images/"
    Path(images_folder).mkdir(parents=True, exist_ok=True)

    
    t1 = time.perf_counter() # başlangıç zamanı
    start_index = 0
    # Run object detection model frame by frame.
    for index, frame in enumerate(frames_json[start_index:], start=start_index):
        # Create a prediction object to store frame info and detections
        predictions = FramePredictions(frame['url'], frame['image_url'], frame['video_name'])
        #print(predictions.image_url)
        # Run detection model
        predictions = detection_model.process(index, predictions,evaluation_server_url)
        # Send model predictions of this frame to the evaluation server
        result = server.send_prediction(predictions)
        response_json = json.loads(result.text)
        if result.status_code == 201:pass
        elif "You do not have permission to perform this action." in response_json["detail"]: # dakikada 80 limiti aşılmışsa
            t2 = time.perf_counter()
            waitTime = 61 - (t1-t2)%60
            time.sleep(waitTime) # 60 saniyeden kalan vakit kadar bekle
            print(f"dakikada 80 frame aşıldı, bekleniliyor... {waitTime} saniye")
            result = server.send_prediction(predictions) # tekrar gönder
            t1 = time.perf_counter() # t1 zamanını yenile

if __name__ == '__main__':
    run()
