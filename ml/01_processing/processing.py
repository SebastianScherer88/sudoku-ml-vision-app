import numpy as np
import pandas as pd
from PIL import Image

from typing import Union, List, Tuple
from pydantic import BaseModel
from pathlib import Path
import random

import os
import shutil

from s3fs import S3FileSystem
from boto3 import s3
import tensorflow as tf

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

def download_and_extract_rar_image_file(clean_up: bool = True) -> Union[Path,str]:
    '''
    Unzips the source .tar file in the specified s3 directory and saves the unzipped folder in the same location.
    Returns the full s3 path to the tf records data file containing the archive content.
    '''

    create_local_temp_dir()

    # download .rar data archive from s3 into local dir if not already there
    rar_source_image_data_local = os.path.join(LOCAL_TEMP_DIR,S3_CELL_DIGIT_CLASSIFICATION_SOURCE_FILE)

    s3_file_system = S3FileSystem()

    if not os.path.isfile(rar_source_image_data_local):
        rar_source_image_data_s3 = f'{S3_CELL_DIGIT_CLASSIFICATION_SOURCE_DIR}/{S3_CELL_DIGIT_CLASSIFICATION_SOURCE_FILE}'
    
        s3_file_system.download(rpath=rar_source_image_data_s3,lpath=rar_source_image_data_local)
        logger.info(f'Downloaded image .rar data file from {rar_source_image_data_s3} into {rar_source_image_data_local}')
        print(f'Downloaded image .rar data file from {rar_source_image_data_s3} into {rar_source_image_data_local}')

    # # extract data into local ./temp/10000 dir
    os.system(f'cd {LOCAL_TEMP_DIR}; 7z x {S3_CELL_DIGIT_CLASSIFICATION_SOURCE_FILE}')
    
    # assemble tf dataset from image data directory structure
    extracted_image_classifciation_data_dir = S3_CELL_DIGIT_CLASSIFICATION_SOURCE_FILE.replace('.rar','')
    local_image_data_dir = os.path.join(LOCAL_TEMP_DIR,extracted_image_classifciation_data_dir)
    logger.info(f'Extracted image .rar data file from {rar_source_image_data_local} into {local_image_data_dir}')
    print(f'Extracted image .rar data file from {rar_source_image_data_local} into {local_image_data_dir}')
    
    tf_train = tf.keras.utils.image_dataset_from_directory(
        directory=local_image_data_dir,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=IMAGE_RECORD_DATASETS_BATCH_SIZE,
        image_size=IMAGE_RECORD_DATASETS_DIMENSION,
        validation_split=0.2,
        subset="training",
        seed=123,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )
    
    tf_validate = tf.keras.utils.image_dataset_from_directory(
        directory=local_image_data_dir,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=IMAGE_RECORD_DATASETS_BATCH_SIZE,
        image_size=IMAGE_RECORD_DATASETS_DIMENSION,
        validation_split=0.2,
        subset="validation",
        seed=123,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )
    
    # export tf datasets locally
    local_tf_train = os.path.join(LOCAL_TEMP_DIR,S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN)
    tf_train.save(local_tf_train)
    logger.info(f'Built tensorflow records data set and exported locally to {local_tf_train}.')
    print(f'Built tensorflow records data set and exported locally to {local_tf_train}.')
    
    local_tf_validate = os.path.join(LOCAL_TEMP_DIR,S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE)
    tf_validate.save(local_tf_validate)
    logger.info(f'Built tensorflow records data set and exported locally to {local_tf_validate}.')
    print(f'Built tensorflow records data set and exported locally to {local_tf_validate}.')


    # upload local tf datasets back to s3
    s3_tf_train = f'{S3_CELL_DIGIT_CLASSIFICATION_TF_DIR}/{S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN}'
    s3_file_system.put(lpath=local_tf_train, 
                       rpath=s3_tf_train,
                       recursive=True)
    
    logger.info(f'Uploaded tensorflow records data set and exported to {s3_tf_train}.')
    print(f'Uploaded tensorflow records data set and exported to {s3_tf_train}.')
    
    # upload extracted data back to s3
    s3_tf_validate = f'{S3_CELL_DIGIT_CLASSIFICATION_TF_DIR}/{S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE}'
    s3_file_system.put(lpath=local_tf_train, 
                       rpath=s3_tf_validate,
                       recursive=True)
    
    logger.info(f'Uploaded tensorflow records data set and exported to {s3_tf_validate}.')
    print(f'Uploaded tensorflow records data set and exported to {s3_tf_validate}.')

    if clean_up:
        clean_up_local_temp_dir()

    return (s3_tf_train,s3_tf_validate)


def get_source_image_inventory(digit_range: List[int] = [0,1,2,3,4,5,6,7,8,9]) -> pd.DataFrame:
    '''
    Creates dataframe of source image meta data holding the 
    - digit,
    - filepath,
    - filename
    - rotation_angle 
    as per the  ImageFile model.
    '''

    images = []

    for digit in digit_range:
        digit_dir = os.path.join(IMAGE_SOURCE_DATA_DIR,str(digit))

        # collect file names & paths
        source_image_file_names = os.listdir(digit_dir)
        source_image_file_paths = [os.path.join(digit_dir,source_image_file) for source_image_file in source_image_file_names]
        rotation_angles = []

        # collect image models
        digit_images = [ImageFile(
                            digit=digit,
                            file_name=source_image_file_name,
                            file_path=source_image_file_path
                        ) for source_image_file_name, source_image_file_path in zip(source_image_file_names,source_image_file_paths)]

        images.extend(digit_images)

    return pd.DataFrame([image.dict() for image in images])

def create_blank_image_from_image(image: Image) -> Image:
    '''
    Takes a real digit source image, randomly picks a spot to infer a background color and creates a blank image of the same size.
    '''

    image = Image.fromarray((image * 255).astype(np.uint8))
    
    image_width, image_height = image.width, image.height

    image_sample_spot = (random.choice(range(image_width)), random.choice(range(image_height)))
    image_spot_color = image.getpixel(image_sample_spot)

    blank_image = Image.new(mode='RGB', size=(image_width, image_height),color=image_spot_color)
    

    return blank_image

def create_blank_images_tf_data(perc_blank_images_train: float = 0.1,
                                perc_blank_images_validate: float = 0.1,
                                clean_up: bool = True) -> pd.DataFrame:
    '''
    Creates and uploads train and validate tf data sets of specified size to s3.
    '''

    s3_blank_train_tf = f'{S3_CELL_DIGIT_CLASSIFICATION_BLANK_TF_DIR}/{S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN_BLANK}'
    s3_blank_avlidate_tf = f'{S3_CELL_DIGIT_CLASSIFICATION_BLANK_TF_DIR}/{S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE_BLANK}'
        
    for data_split in (S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN, S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE):
        
        # load local data exported in previous step
        data_tf = tf.data.Dataset.load(os.path.join(LOCAL_TEMP_DIR,data_split))
        
        for batch in data_tf:
        

    # blank_images = []
    
    # counter = 0

    # if not os.path.exists(blank_digit_dir):
    #     os.makedirs(blank_digit_dir)

    # for index, image_file in source_image_inventory.iterrows():
    #     image = Image.open(image_file['file_path'])
    #     blank_image = create_blank_image(image)
    #     blank_image_file_name = f"blank_{image_file['file_name']}"
    #     blank_image_file_path = os.path.join(blank_digit_dir,blank_image_file_name)
    #     blank_image.save(blank_image_file_path)

    #     blank_images.append(ImageFile(digit=BLANK_DIGIT,file_name=blank_image_file_name,file_path=blank_image_file_path))

    #     counter += 1

    #     if counter >= N_BLANK_IMAGES:
    #         break

    if clean_up:
        clean_up_local_temp_dir()

    return pd.DataFrame([blank_image.dict() for blank_image in blank_images])

def rotate_image(digit_image: Image,
                rotation_angle: float = 10) -> Image:
    '''
    Rotates a digit source image by the designated angle counter-clockwise and returns a cropped image of the original size.
    '''

    rotated_image = digit_image.rotate(rotation_angle)

    return rotated_image

def create_rotated_images(source_image_inventory: pd.DataFrame) -> pd.DataFrame:

    rotated_dir = os.path.join([IMAGE_SYNTHETIC_DATA_DIR,ROTATED_IMAGE_DIR])

    if not os.path.exists(rotated_dir):
        os.makedirs(rotated_dir)

    rotated_images = []

    for digit in [0,1,2,3,4,5,6,7,8,9]:
        counter = 0
        digit_image_inventory = source_image_inventory.loc[source_image_inventory.digit == digit]

        for index, image_file in digit_image_inventory.iterrows():
            image = Image.open(image_file['file_path'])
            rotation_angle = round(random.uniform(*ANGLE_RANGE),1)
            rotated_image = rotate_image(image,rotation_angle=rotation_angle)
            rotated_image_file_name = f"rotated_{rotation_angle}_{image_file['file_name']}"
            rotated_image_file_path = os.path.join(rotated_dir,rotated_image_file_name)
            rotated_image.save(rotated_image_file_path)

            rotated_images.append(ImageFile(digit=digit,file_name=rotated_image_file_name,file_path=rotated_image_file_path,rotation_angle=rotation_angle))

            counter += 1

            if counter >= N_ROTATED_IMAGES_PER_DIGIT:
                break

    return pd.DataFrame([rotated_image.dict() for rotated_image in rotated_images])


def main():
    
    download_and_extract_rar_image_file(clean_up=True)

    # # get all image data inventory
    # source_image_inventory = get_source_image_inventory()

    # # add blanks as pseudo digit 10 in new source subdirectory
    # blank_image_inventory = create_blank_images(source_image_inventory)

    # # add rotated images for digit 1-9 in existing subdirectories
    # rotated_image_inventory = create_rotated_images(source_image_inventory)

    # all_images_inventory = pd.concat([source_image_inventory,blank_image_inventory,rotated_image_inventory]).reset_index()
    # all_images_inventory.to_csv(os.path.join(IMAGE_SYNTHETIC_DATA_DIR,'all_image_inventory.csv'),index=False)


if __name__ == '__main__':
    main()