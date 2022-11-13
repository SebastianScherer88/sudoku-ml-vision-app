import numpy as np
import pandas as pd
from PIL import Image

from typing import Union, List, Tuple
from pydantic import BaseModel
from pathlib import Path
import random

import os

from model.settings import (
    IMAGE_SOURCE_DATA_DIR, IMAGE_SYNTHETIC_DATA_DIR, RANDOM_SEED,
    BLANK_IMAGE_DIR, N_BLANK_IMAGES, BLANK_DIGIT,
    ROTATED_IMAGE_DIR, N_ROTATED_IMAGES_PER_DIGIT, ANGLE_RANGE
)

random.seed(RANDOM_SEED)

class ImageFile(BaseModel):
    digit: int
    file_name: str
    file_path: Path
    rotation_angle: int = 0


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

def create_blank_image(digit_image: Image) -> Image:
    '''
    Takes a real digit source image, randomly picks a spot to infer a background color and creates a blank image of the same size.
    '''

    image_width, image_height = digit_image.width, digit_image.height

    image_sample_spot = (random.choice(range(image_width)), random.choice(range(image_height)))
    image_spot_color = digit_image.getpixel(image_sample_spot)

    blank_image = Image.new(mode='RGB', size=(image_width, image_height),color=image_spot_color)

    return blank_image

def create_blank_images(source_image_inventory: pd.DataFrame) -> pd.DataFrame:
    '''
    Creats blank image files and exports them to the same image source data dir as the original image files, in an extra directory named '10' encoding the 'blank' pseudo digit.
    '''

    blank_digit_dir = os.path.join(IMAGE_SYNTHETIC_DATA_DIR,BLANK_IMAGE_DIR)

    blank_images = []

    counter = 0

    if not os.path.exists(blank_digit_dir):
        os.makedirs(blank_digit_dir)

    for index, image_file in source_image_inventory.iterrows():
        image = Image.open(image_file['file_path'])
        blank_image = create_blank_image(image)
        blank_image_file_name = f"blank_{image_file['file_name']}"
        blank_image_file_path = os.path.join(blank_digit_dir,blank_image_file_name)
        blank_image.save(blank_image_file_path)

        blank_images.append(ImageFile(digit=BLANK_DIGIT,file_name=blank_image_file_name,file_path=blank_image_file_path))

        counter += 1

        if counter >= N_BLANK_IMAGES:
            break

    return pd.DataFrame([blank_image.dict() for blank_image in blank_images])

def rotate_image(digit_image: Image,
                rotation_angle: float = 10) -> Image:
    '''
    Rotates a digit source image by the designated angle counter-clockwise and returns a cropped image of the original size.
    '''

    rotated_image = digit_image.rotate(rotation_angle)

    return rotated_image

def create_rotated_images(source_image_inventory: pd.DataFrame) -> pd.DataFrame:

    rotated_dir = os.path.join(IMAGE_SYNTHETIC_DATA_DIR,ROTATED_IMAGE_DIR)

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

    # get all image data inventory
    source_image_inventory = get_source_image_inventory()

    # add blanks as pseudo digit 10 in new source subdirectory
    blank_image_inventory = create_blank_images(source_image_inventory)

    # add rotated images for digit 1-9 in existing subdirectories
    rotated_image_inventory = create_rotated_images(source_image_inventory)

    all_images_inventory = pd.concat([source_image_inventory,blank_image_inventory,rotated_image_inventory]).reset_index()
    all_images_inventory.to_csv(os.path.join(IMAGE_SYNTHETIC_DATA_DIR,'all_image_inventory.csv'),index=False)


main()