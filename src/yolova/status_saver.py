
import json
import os
from pathlib import Path
from os.path import exists

class SendedPrediction:
    def __init__(self, index: int, sendTime: float) -> None:
        self.index = index
        self.sendTime = sendTime

    def toJson(self):
        return {'index': self.index, 'sendTime': self.sendTime}

    @staticmethod
    def fromJson(json):
        index = json['index']
        sendTime = json['sendTime']
        return SendedPrediction(index, sendTime)


class PredictStatusSaver:
    def __init__(self) -> None:
        self.save_folder = "./_predict_status/"
        self.save_file_name = "last.json"
        Path(self.save_folder).mkdir(parents=True, exist_ok=True)
        self.sended_predictions:list[SendedPrediction]=[]
        self.get_saved_data()
    
    def isSended(self, index:int):
        return index in [x.index for x in self.sended_predictions]

    def addSendedPrediction(self, sendedPrediction: SendedPrediction):
        # self.save_data["last_index"] = index
        
        if sendedPrediction.index in [x.index for x in self.sended_predictions]:
            return
        if "sended_predictions" not in self.save_data:
            self.save_data["sended_predictions"]=[]
        self.save_data["sended_predictions"].append(sendedPrediction.toJson())
        with open(self.save_folder+self.save_file_name, 'w', encoding='utf-8') as f:
            json.dump(self.save_data, f, ensure_ascii=False, indent=4)
        self.get_saved_data()

    def get_saved_data(self):
        self.save_data = {}
        save_file= os.path.join(self.save_folder, self.save_file_name)
        if not exists(save_file):
            self.save_data = {
                'sended_predictions': []
            }
            with open(self.save_folder+self.save_file_name, 'w', encoding='utf-8') as f:
                json.dump(self.save_data, f, ensure_ascii=False, indent=4)
        else:   
            with open(self.save_folder+self.save_file_name, "r+") as json_file:
                self.save_data = json.load(json_file)
        self.get_sended_predictions()

    def get_sended_predictions(self) -> list[SendedPrediction]:
        if "sended_predictions" not in self.save_data:
            self.sended_predictions=[]
            return
        self.sended_predictions= list(map(lambda x: SendedPrediction.fromJson(x), self.save_data["sended_predictions"]))

    def clear(self):
        data = {
            'sended_predictions': []
        }
        with open(self.save_folder+self.save_file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        self.get_saved_data()
