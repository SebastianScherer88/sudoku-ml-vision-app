import tensorflow as tf

import os
from ml.settings import (
    LOCAL_TEMP_DIR,
    S3_DATA_BUCKET,
    S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN_ALL,
    S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE_ALL,
    IMAGE_RECORD_DATASETS_DIMENSION,
    IMAGE_CHANNEL_N,
    MODEL_DIR,
    S3_MODEL_REGISTER
)
from typing import Tuple
from keras.losses import categorical_crossentropy as xentropy
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.optimizers import Adam

import boto3
import json

from ml.processing.build_dataset import normalize_image
from utils.s3 import upload_directory

from datetime import datetime
from ml.settings import logger
import pandas as pd

export_to_s3: bool = True

def get_datasets():
    
    local_path_train = os.path.join(LOCAL_TEMP_DIR,S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN_ALL)
    local_path_validate = os.path.join(LOCAL_TEMP_DIR,S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE_ALL)
    
    train = tf.data.Dataset.load(local_path_train)
    validate = tf.data.Dataset.load(local_path_validate)
    
    return train, validate

def get_compiled_model(n_channels_conv1: int = 10,
                       kernel_dim_conv1: Tuple[int,int] = (3,3),
                       n_channels_conv2: int = 20,
                       kernel_dim_conv2: Tuple[int,int] = (3,3),
                       n_channels_conv3: int = 40,
                       kernel_dim_conv3: Tuple[int,int] = (3,3),
                       n_dense_1: int = 100,
                       learning_rate: float = 0.01,
                       *args,
                       **kwargs):
    
    image_dims = (IMAGE_RECORD_DATASETS_DIMENSION[0],IMAGE_RECORD_DATASETS_DIMENSION[1],IMAGE_CHANNEL_N)
    model = Sequential()
    #add model layers
    model.add(Conv2D(n_channels_conv1, kernel_dim_conv1, activation='relu', input_shape=image_dims))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(n_channels_conv2, kernel_dim_conv2, activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(n_channels_conv3, kernel_dim_conv3, activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(n_dense_1, activation='relu'))
    model.add(Dense(10, activation = 'softmax'))
    
    logger.info(model.summary())
    
    opt = Adam(learning_rate=learning_rate)
    
    model.compile(optimizer= opt,
                  loss=xentropy,
                  metrics=['accuracy'])
    
    return model

def build_model(model,
                train,
                validate,
                export_to_s3: bool = True,
                **fit_kwargs):
    
    history = model.fit(train,
                        validation_data=validate,
                        **fit_kwargs)
    
    history_nice = pd.DataFrame(history.history)

    # export model locally    
    model_version = datetime.now().strftime(format="%Y_%m_%d__%H_%M_%S")

    model_version_dir = os.path.join(LOCAL_TEMP_DIR,MODEL_DIR,model_version)
    os.makedirs(model_version_dir)
    
    model_path = os.path.join(model_version_dir,'model')
    model.save(model_path)
    
    history_path = os.path.join(model_version_dir,'history.csv')
    history_nice.to_csv(history_path,index=False)
    
    fit_params_path = os.path.join(model_version_dir,'fit_params.json')
    with open(fit_params_path,'w') as fit_params_file:
        json.dump(fit_kwargs, fit_params_file)
        
    # upload model export directory to s3
    if export_to_s3:        
        model_version_prefix = f'{S3_MODEL_REGISTER}/{model_version}'
        
        # upload all model related files
        upload_directory(model_version_dir,S3_DATA_BUCKET,model_version_prefix)
    
    return model, history_nice

def main():
    
    train, validate = get_datasets()
    
    model = get_compiled_model(learning_rate=0.01)
    
    trained_model, history = build_model(model,
                                         train, 
                                         validate,
                                         export_to_s3=export_to_s3,
                                         epochs=1,
                                         batch_size=200)
    
    
    logger.info(f'Train history: {history}')
    
if __name__ == '__main__':
    main()
    
    