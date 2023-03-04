import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from flask_restful import Resource
from datapreparation.extractor import PoseExtractor

class data_prep:
    def __init__(self) -> None:
        pass

    def process_data(self,images_dir, csv_path, pose_model, body_dict,opration):
        extractor = PoseExtractor(pose_model, body_dict)
        data = {key: [] for key in body_dict.keys()}
        data['label'] = []
        for entry in os.scandir(images_dir):
            if not entry.is_dir():
                continue
            label = entry.name
            for file in tqdm(os.listdir(entry.path)):
                cap = cv2.VideoCapture(os.path.join(entry.path, file))
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        pose_data = extractor.extract([frame])[0]
                        for key in data.keys():
                            if key!='label':
                                data[key].append(int(pose_data[body_dict[key]]))
                        data['label'].append(label)
                    else:
                        break
                cap.release()
        if opration=='train':
            df = pd.DataFrame(data)
            df_present=pd.read_csv(csv_path)
            concate_data = pd.concat([df,df_present])
            concate_data.to_csv(csv_path, index=False)
        else:
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
        return " The data has been processed" 

    def data_trained(self,csv):
        df_data = pd.read_csv(csv)
        list_unq=df_data['label'].unique()
        return list_unq