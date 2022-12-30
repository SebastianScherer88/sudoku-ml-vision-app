import kserve
from typing import Dict
from PIL import Image
import base64
import io

from keras.models import load_model
from utils.s3 import download_directory
from ml.processing.build_dataset import normalize_image
import os
import numpy as np
import json

# pod/container variables
S3_MODEL_REGISTRY_BUCKET = os.environ.get('S3_MODEL_REGISTRY_BUCKET','bettmensch88-aws-dev-bucket')
S3_MODEL_REGISTRY_PATH = os.environ.get('S3_MODEL_REGISTRY_ENTRY','sudoku-ml-vision/ml_models')
LOCAL_MODEL_REGISTRY_PATH = os.environ.get('LOCAL_MODEL_REGISTRY_PATH','./temp')
MODEL_VERSION = os.environ.get('MODEL_VERSION') # the only one thats required

class DigitClassifier(kserve.Model):
    def __init__(self, s3_model_registry_entry: str):
        super().__init__('digit-classifier')
        self.name = 'digit-classifier'
        self.model_registry_entry = s3_model_registry_entry
        self.load(s3_model_registry_entry)
        self.model = None
        self.ready = False

    def load(self):
        
        # download model assets
        download_directory(S3_MODEL_REGISTRY_BUCKET, f'{S3_MODEL_REGISTRY_PATH}/{MODEL_VERSION}', LOCAL_MODEL_REGISTRY_PATH)
        
        local_model_path = os.path.join(LOCAL_MODEL_REGISTRY_PATH,MODEL_VERSION,'model')
        self.model = load_model(local_model_path)
        
        local_index_label_map_path = os.path.join(LOCAL_MODEL_REGISTRY_PATH,MODEL_VERSION,'index_label_map.json')
        self.index_label_map = json.load(local_index_label_map_path)
        
        self.ready = True
        
    def preprocess(self, payload: Dict) -> np.array:
        
        image_data = []
        
        for request_instance in payload['instances']:
            encoded_image_data = request_instance['image']['b64']
            
            decoded_image_data = base64.b64decode(encoded_image_data)
            resized_image_data = np.asarray(Image.open(io.BytesIO(decoded_image_data)).resize((100,100)))
            image_data.append(resized_image_data)

        image_data_array = np.concatenate(image_data, axis=1)
        
        return image_data_array

    def predict(self, image_array: np.array) -> Dict:
        
        predictions = self.model.predict(image_array)

        return predictions
    
    def postprocess(self, predictions: np.array, return_labels: bool = True) -> Dict:
        
        if return_labels:
            max_indices = predictions.argmax(axis=0).tolist()
            prediction_payload = [self.index_label_map[str(max_index)] for max_index in max_indices]
        else:
            prediction_payload = predictions.tolist()
            
        response = {'predictions': prediction_payload}
        
        return response

if __name__ == "__main__":
    model = DigitClassifier("digit-classifier")
    model.load()
    kserve.ModelServer(workers=1).start([model])