import base64
import io
import json
import os
from typing import Dict

import kserve
import numpy as np
from keras.models import load_model
from PIL import Image

from image_processing.processing.build_model_dataset import normalize_image
from ml_digit_classification_utils import (
    DigitImage,
    decode_image_file_from_http
)
from utils.s3 import download_directory

# pod/container variables
S3_MODEL_REGISTRY_BUCKET = os.environ.get('S3_MODEL_REGISTRY_BUCKET','bettmensch88-aws-dev-bucket')
S3_MODEL_REGISTRY_PATH = os.environ.get('S3_MODEL_REGISTRY_ENTRY','sudoku-ml-vision/ml_models')
LOCAL_MODEL_REGISTRY_PATH = os.environ.get('LOCAL_MODEL_REGISTRY_PATH','./temp')
MODEL_VERSION = os.environ.get('MODEL_VERSION','2022_12_30__20_55_45') # the only one thats required

class DigitClassifier(kserve.Model):
    def __init__(self, s3_model_registry_entry: str):
        super().__init__('digit-classifier')
        self.name = 'digit-classifier'
        self.model_registry_entry = s3_model_registry_entry
        self.model = None
        self.index_label_map = None
        self.ready = False

    def load(self):
        
        # download model assets
        download_directory(S3_MODEL_REGISTRY_BUCKET, f'{S3_MODEL_REGISTRY_PATH}/{MODEL_VERSION}', LOCAL_MODEL_REGISTRY_PATH)
        
        local_model_path = os.path.join(LOCAL_MODEL_REGISTRY_PATH,MODEL_VERSION,'model')
        self.model = load_model(local_model_path)
        
        local_index_label_map_path = os.path.join(LOCAL_MODEL_REGISTRY_PATH,MODEL_VERSION,'index_label_map.json')
        with open(local_index_label_map_path,'r') as index_label_map_file:
            self.index_label_map = json.load(index_label_map_file)
        
        self.ready = True
        
    def image_transform(self,image):
        
        return image.resize((100,100))
        
    def preprocess(self, payload: DigitImage, *args, **kwargs) -> np.array:
        
        validated_payload = DigitImage(**payload)
        
        image_data = []
                
        for encoded_image_data in validated_payload.instances:
            
            image = decode_image_file_from_http(encoded_image_data)
            resized_image = self.image_transform(image)
            resized_image_data = np.asarray(resized_image)
            image_data.append(resized_image_data)

        image_data_array = np.stack(image_data, axis=0)
        
        return image_data_array

    def predict(self, image_array: np.array, *args, **kwargs) -> Dict:
        
        predictions = self.model.predict(image_array)

        return predictions
    
    def postprocess(self, predictions: np.array, return_labels: bool = True, *args, **kwargs) -> Dict:
        
        if return_labels:
            max_indices = predictions.argmax(axis=-1).tolist()
            prediction_payload = [self.index_label_map[str(max_index)] for max_index in max_indices]
        else:
            prediction_payload = predictions.tolist()
            
        response = {'predictions': prediction_payload}
        
        return response

if __name__ == "__main__":
    model = DigitClassifier("digit-classifier")
    model.load()
    kserve.ModelServer().start([model])