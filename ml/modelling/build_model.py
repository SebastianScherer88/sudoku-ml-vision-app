import tensorflow as tf
import tensorflow_addons as tfa

import os
from ml.settings import (
    LOCAL_TEMP_DIR,
    S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN_ALL,
    S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE_ALL,
    IMAGE_RECORD_DATASETS_DIMENSION,
    IMAGE_CHANNEL_N,
    MODEL_DIR
)
from keras.losses import categorical_crossentropy as xentropy
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.optimizers import Adam

from ml.processing.build_dataset import normalize_image

from datetime import datetime
from ml.settings import logger
import pandas as pd

def get_datasets():
    
    local_path_train = os.path.join(LOCAL_TEMP_DIR,S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN_ALL)
    local_path_validate = os.path.join(LOCAL_TEMP_DIR,S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE_ALL)
    
    train = tf.data.Dataset.load(local_path_train)
    validate = tf.data.Dataset.load(local_path_validate)
    
    return train, validate

def get_compiled_model(learning_rate: float = 0.01):
    
    image_dims = (IMAGE_RECORD_DATASETS_DIMENSION[0],IMAGE_RECORD_DATASETS_DIMENSION[1],IMAGE_CHANNEL_N)
    model = Sequential()
    #add model layers
    model.add(Conv2D(10, (3, 3), activation='relu', input_shape=image_dims))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(20, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
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
                n_epochs: int = 3,
                export_local: bool = True,
                **fit_kwargs):
    
    history = model.fit(train,
                        epochs=n_epochs, 
                        validation_data=validate,
                        **fit_kwargs)
    
    history_nice = pd.DataFrame(history.history)
    
    if export_local:
        
        model_version = datetime.now().strftime(format="%Y_%m_%d__%H_%M_%S")
        model_version_dir = os.path.join(LOCAL_TEMP_DIR,MODEL_DIR,model_version)
        os.makedirs(model_version_dir)
        
        model_path = os.path.join(model_version_dir,'model')
        model.save(model_path)
        
        history_path = os.path.join(model_version_dir,'history')
        history_nice.to_csv(history_path,index=False)
    
    return model, history_nice

def main():
    
    train, validate = get_datasets()
    
    model = get_compiled_model(learning_rate=0.01)
    
    trained_model, history = build_model(model, train, validate, n_epochs=7)
    
    logger.info(f'Train history: {history}')
    
if __name__ == '__main__':
    main()
    
    