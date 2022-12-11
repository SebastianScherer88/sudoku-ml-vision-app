import numpy as np
import pandas as pd
from PIL import Image

from typing import Union, List, Tuple
from pydantic import BaseModel
from pathlib import Path
import random
import matplotlib.pyplot as plt

import os
import shutil

from s3fs import S3FileSystem
from boto3 import s3
import tensorflow as tf
import tensorflow_addons as tfa

from logging import getLogger

from ml.settings import (
    LOG_LEVEL,
    LOCAL_TEMP_DIR,
    S3_CELL_DIGIT_CLASSIFICATION_SOURCE_DIR, 
    S3_CELL_DIGIT_CLASSIFICATION_SOURCE_FILE,
    IMAGE_RECORD_DATASETS_DIMENSION,
    IMAGE_RECORD_DATASETS_BATCH_SIZE,
    S3_CELL_DIGIT_CLASSIFICATION_TF_DIR,
    S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN,
    S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE,
    S3_CELL_DIGIT_CLASSIFICATION_ROTATED_TF_DIR,
    S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN_ROTATED,
    S3_CELL_DIGIT_CLASSIFICATION_BLANK_TF_DIR,
    S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN_BLANK,
    S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE_BLANK,
    IMAGE_SOURCE_DATA_DIR, 
    IMAGE_SYNTHETIC_DATA_DIR, 
    RANDOM_SEED,
    BLANK_IMAGE_DIR,
    N_BLANK_IMAGES, BLANK_DIGIT,
    ROTATED_IMAGE_DIR, 
    N_ROTATED_IMAGES_PER_DIGIT, 
    ANGLE_RANGE
)

random.seed(RANDOM_SEED)

logger = getLogger('processing')
logger.setLevel(LOG_LEVEL)

class ImageFile(BaseModel):
    digit: int
    file_name: str
    file_path: Path
    rotation_angle: int = 0


def create_local_temp_dir():
    
    # create temporary local dir
    if not os.path.exists(LOCAL_TEMP_DIR):
        os.mkdir(LOCAL_TEMP_DIR)
        logger.info(f'Created local temp directory {LOCAL_TEMP_DIR}')
        print(f'Created local temp directory {LOCAL_TEMP_DIR}')
        
def clean_up_local_temp_dir():
    
    # delete local temp dir and content
    shutil.rmtree(LOCAL_TEMP_DIR, ignore_errors=True)
    logger.info(f'Removed local temporary directory {LOCAL_TEMP_DIR} and all its contents.')
    print(f'Removed local temporary directory {LOCAL_TEMP_DIR} and all its contents.')

def download_and_extract_rar_image_file() -> Union[Path,str]:
    '''
    Unzips the source .tar file in the specified s3 directory and saves the unzipped folder in the same location.
    Returns the full s3 path to the tf records data file containing the archive content.
    '''

    # download .rar data archive from s3 into local dir if not already there
    rar_source_image_data_local = os.path.join(LOCAL_TEMP_DIR,S3_CELL_DIGIT_CLASSIFICATION_SOURCE_FILE)

    s3_file_system = S3FileSystem()

    rar_source_image_data_s3 = f'{S3_CELL_DIGIT_CLASSIFICATION_SOURCE_DIR}/{S3_CELL_DIGIT_CLASSIFICATION_SOURCE_FILE}'

    s3_file_system.download(rpath=rar_source_image_data_s3,lpath=rar_source_image_data_local)
    logger.info(f'Downloaded image .rar data file from {rar_source_image_data_s3} into {rar_source_image_data_local}')

    # # extract data into local ./temp/10000 dir
    os.system(f'cd {LOCAL_TEMP_DIR}; 7z x {S3_CELL_DIGIT_CLASSIFICATION_SOURCE_FILE}')
    
    # assemble tf dataset from image data directory structure
    extracted_image_classifciation_data_dir = S3_CELL_DIGIT_CLASSIFICATION_SOURCE_FILE.replace('.rar','')
    local_image_data_dir = os.path.join(LOCAL_TEMP_DIR,extracted_image_classifciation_data_dir)
    logger.info(f'Extracted image .rar data file from {rar_source_image_data_local} into {local_image_data_dir}')
    
    return local_image_data_dir
    
def create_train_test(local_image_data_dir: Union[str,Path],
                      validation_split: float = 0.2,
                      export_local: bool = True):
    
    tf_train = tf.keras.utils.image_dataset_from_directory(
        directory=local_image_data_dir,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=None,#IMAGE_RECORD_DATASETS_BATCH_SIZE,
        image_size=IMAGE_RECORD_DATASETS_DIMENSION,
        validation_split=validation_split,
        subset="training",
        seed=123,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    ).filter(lambda image,label: label != 0)
    
    tf_validate = tf.keras.utils.image_dataset_from_directory(
        directory=local_image_data_dir,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=None,#IMAGE_RECORD_DATASETS_BATCH_SIZE,
        image_size=IMAGE_RECORD_DATASETS_DIMENSION,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    ).filter(lambda image,label: label != 0)
    
    if export_local:
        # export tf datasets locally
        local_tf_train = os.path.join(LOCAL_TEMP_DIR,S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN)
        tf_train.save(local_tf_train)
        logger.info(f'Built tensorflow records data set and exported locally to {local_tf_train}.')
        
        local_tf_validate = os.path.join(LOCAL_TEMP_DIR,S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE)
        tf_validate.save(local_tf_validate)
        logger.info(f'Built tensorflow records data set and exported locally to {local_tf_validate}.')
    
    return tf_train, tf_validate

def rotate_images(image_tensor_batch, image_label_batch):
    
    rotation_angle = np.pi / random.uniform(3,12)
    
    rotated_image_tensor_batch = tfa.image.rotate(image_tensor_batch,
                                                  tf.constant(rotation_angle))
    
    return rotated_image_tensor_batch, image_label_batch


def create_rotated_images(train_dataset,
                          export_local: bool = True) -> str:
    
    rotated_dataset = train_dataset.map(rotate_images)
    
    # export rotated train image data to local
    if export_local:
        tf_train_rotated_image_data_local = os.path.join(LOCAL_TEMP_DIR,S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN_ROTATED)
        rotated_dataset.save(tf_train_rotated_image_data_local)
    
    return rotated_dataset

def blank_images(image_tensor_batch, image_label_batch):
    
    blanked_image_tensor_batch = tfa.image.gaussian_filter2d(image_tensor_batch,
                                                             sigma = 100)
    
    import pdb
    pdb.set_trace()
    
    blanked_image_label_batch = image_label_batch * 0 + 10
    
    return blanked_image_tensor_batch, blanked_image_label_batch

def get_blank_generator(dataset):
    
    def blank_generator():
        
        for image_batch, image_label_batch in dataset.as_numpy_iterator():
        
            n_batch, width, height, n_channel = image_batch.shape

            image_sample_x, image_sample_y = (random.choice(range(width)), random.choice(range(height)))
            
            image_batch_fibre = image_batch[:,image_sample_x, image_sample_y, :]
            image_batch_fibre = np.expand_dims(image_batch_fibre,axis=[1,2])
            
            blank_image_batch_slice = np.concatenate([image_batch_fibre,] * width,axis=1)
            blank_image_batch = np.concatenate([blank_image_batch_slice,] * height,axis=2)
            blanked_image_label_batch = image_label_batch * 0

            yield blank_image_batch, blanked_image_label_batch
    
    return blank_generator

def create_blanked_images(train_dataset,
                          validate_dataset,
                          export_local: bool = True) -> str:
        
    train_blank_generator = get_blank_generator(train_dataset)
    validate_blank_generator = get_blank_generator(validate_dataset)
    
    blanked_train_dataset = tf.data.Dataset.from_generator(train_blank_generator,
                                                           output_signature=(tf.TensorSpec(shape=(None, 100, 100, 3), dtype=tf.float32, name=None), 
                                                                             tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None))
    )
    
    blanked_validate_dataset = tf.data.Dataset.from_generator(validate_blank_generator,
                                                              output_signature=(tf.TensorSpec(shape=(None, 100, 100, 3), dtype=tf.float32, name=None), 
                                                                                tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None))
    )
    
    if export_local:
        # export rotated train image data to local
        tf_train_blanked_image_data_local = os.path.join(LOCAL_TEMP_DIR,S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN_BLANK)
        blanked_train_dataset.save(tf_train_blanked_image_data_local)
        
        tf_validate_blanked_image_data_local = os.path.join(LOCAL_TEMP_DIR,S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE_BLANK)
        blanked_validate_dataset.save(tf_validate_blanked_image_data_local)
    
    return blanked_train_dataset, blanked_validate_dataset


def main():
    
    create_local_temp_dir()
    
    image_data_path = download_and_extract_rar_image_file()
    
    tf_train, tf_validate = create_train_test(image_data_path,
                                              validation_split = 0.2,
                                              export_local = True)
    
    # add rotated images & export local
    train_rotated = create_rotated_images(tf_train,
                                          export_local = True)
    
    # add blank images & export local
    train_blank, validate_blank = create_blanked_images(tf_train,
                                                        tf_validate,
                                                        export_local = True)

if __name__ == '__main__':
    main()