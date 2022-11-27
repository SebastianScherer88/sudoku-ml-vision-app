# -*- coding: utf-8 -*-
"""
Created on Sun May 10 20:06:32 2020

Creates artificial sudoku images with training label arrays to be used for the
convolutional neural network recognizing sudoku initial values.

@author: bettmensch
"""

from PIL import Image, ImageDraw
import os
import numpy as np
from datetime import datetime
import json

digit_folder = os.path.normpath(r'C:\Users\bettmensch\GitReps\sudoku_solver\data\digits\individual_pics\calibri')

digit_image_paths = [os.path.join(digit_folder,image_name) for image_name in os.listdir(digit_folder)]
digit_names = [file_name.replace('.png','') for file_name in os.listdir(digit_folder)]
digit_images = [Image.open(digit_image_path) for digit_image_path in digit_image_paths]

# concatenate images
first = digit_images[0]
second = digit_images[1]

canvas = Image.new('RGB',(first.width + second.width, first.height + second.height))
canvas.paste(first,(0,0))
canvas.paste(second,(first.width,0))
canvas.paste(first,(first.width,first.height))
canvas.paste(second,(0,first.height))

# draw lines
draw = ImageDraw.Draw(canvas)
draw.line((int(canvas.width / 2), 0, int(canvas.width / 2), canvas.height), fill=(100, 100, 100), width=3)
draw.line((0, int(canvas.height / 2), canvas.width, int(canvas.height / 2)), fill=(100, 100, 100), width=3)



def get_digit_images(fonts = ['calibri','times_new_roman','courier_new'],
                     base_path = r'C:\Users\bettmensch\GitReps\sudoku_solver\data\digits\individual_pics'):
    '''Helper function that loads in .png images of digits 1-9 of a specified font.'''
    
    assert os.path.isdir(base_path)

    digit_font_images = {}
    
    for font in fonts:
        digit_file_directory = os.path.join(base_path,font)

        digit_image_paths = [os.path.join(digit_file_directory,image_name) for image_name in os.listdir(digit_file_directory)]
        digit_names = [file_name.replace('.png','') for file_name in os.listdir(digit_file_directory)]
        digit_images = [Image.open(digit_image_path) for digit_image_path in digit_image_paths]
        
        images = dict([(k,v) for k,v in zip(digit_names, digit_images)])
        images['blank'] = Image.new('RGB', (digit_images[0].width, digit_images[0].height),(255,255,255))
        
        digit_font_images[font] = images
    
    return digit_font_images

def get_sudoku_grid(font_images: dict,
                    blank_ratio = 0.1):
    '''Helper function that returns the 81x81 integer array with integers.
    Serves as the response to the convolutional network.'''
    
    assert 0 <= blank_ratio and blank_ratio <= 1
    
    fonts = list(font_images.keys())
    categories = list(font_images[fonts[0]].keys())
    
    assert 'blank' in categories
    
    # get probability array
    probabilities = [(1 - blank_ratio) / (len(categories) - 1)]*len(categories)
    probabilities[categories.index('blank')] = blank_ratio
    
    grid_values = np.random.choice(a = categories,
                                   size = 81,
                                   replace = True,
                                   p = probabilities).reshape(9,9)
    
    grid_fonts = np.random.choice(a = fonts,
                                  size = 81,
                                  replace = True,
                                  p = [1/3] * len(fonts)).reshape(9,9)
    
    return grid_values, grid_fonts

def get_sudoku_image_and_annotation(sudoku_grid_array: np.array,
                                     sudoku_font_array: np.array,
                                     font_images: dict,
                                     image_size = (256,256), # needs to be compatible with mask-rcnn input shapes
                                     bounding_box_jitter = 0.00,
                                     as_array = False):
    '''Helper function that creates and returns an 540x540 image of a sudoku grid according to initial values
    contained in the sudoku_grid_array. Serves as the input to the convolutional network.'''
    
    assert sudoku_grid_array.shape == (9,9)
    
    #get fonts
    fonts = list(font_images.keys())
    
    # get individual dimensions
    width, height = font_images[fonts[0]]['blank'].width, font_images[fonts[0]]['blank'].height
    
    # --- build image
    sudoku_image = Image.new('RGB',(width * 9, height * 9))
    
    # paste the digit images
    for row_index in range(9):
        for column_index in range(9):
            
            random_font = sudoku_font_array[row_index,column_index]
            random_value = sudoku_grid_array[row_index, column_index]
            
            sudoku_image.paste(font_images[random_font][random_value],
                              (column_index*width,row_index*height))
            
    # paste the vertical and horizontal dividing lines
    line_drawer = ImageDraw.Draw(sudoku_image)

    for row_index in range(10):
        line_drawer.line((0, row_index*height, width * 9, row_index*height), fill=(100, 100, 100), width=3)
    
    for column_index in range(10):
        line_drawer.line((column_index*width, 0, column_index*width, height * 9), fill=(100, 100, 100), width=3)
    
    sudoku_image = sudoku_image.resize(image_size)
        
    if as_array:
        # convert to 2-dim grey scale array representation
        sudoku_image = np.asarray(sudoku_image, dtype="int32" ).mean(axis = -1)
        
    # --- build annotation
    annotation = {'filename': None,
                  'regions': []}
    
    index = 0
    x_noise_bound = image_size[0] * bounding_box_jitter
    y_noise_bound = image_size[1] * bounding_box_jitter
    
    #print(x_noise_bound)
    
    resized_width = round(width * image_size[0] / (width * 9))
    resized_height = round(height * image_size[1] / (height * 9))
    x_resize_factor, y_resize_factor = image_size[0] / (width*9), image_size[1] / (width*9)
    
    #print(resized_width)
    
    for row_index in range(9):
        for column_index in range(9):
            
            x,y,value = round(column_index*width*x_resize_factor), round(row_index*height*y_resize_factor), sudoku_grid_array[row_index, column_index]
            x_noises = np.random.uniform(-x_noise_bound,x_noise_bound,2).astype(int)
            y_noises = np.random.uniform(-y_noise_bound,y_noise_bound,2).astype(int)
            
            #print(x_noises)
            
            # annotate bounding box parameters with some optional noise and class
            segment = {'shape_attributes':{'name':'rect',
                                'x':max(0,min(x + int(x_noises[0]),image_size[0]-1)),
                                'y':max(0,min(y + int(y_noises[0]),image_size[1]-1)),
                                'width':max(0,min(resized_width + int(x_noises[1]),image_size[0]-1)),
                                'height':max(0,min(resized_height + int(y_noises[1]),image_size[1]-1))},
             'region_attributes':{'grid_cell': str(value)}}
            
            annotation['regions'].append(segment)
            
            #print(annotation['regions'][index])
            
            index += 1
            
    return sudoku_image, annotation

def create_sudoku_data_set(n = 500,
                           blank_percentage = 0.1,
                           font = 'calibri',
                           digit_base_path = r'C:\Users\bettmensch\GitReps\sudoku_solver\data\digits\individual_pics',
                           data_set_outdir = r'C:\Users\bettmensch\GitReps\Mask_RCNN\datasets\sudoku',
                           data_set_out_name = 'train'):
    '''Helper function that creates inputs and response to the convolutional network meant to convert an image of a started sudoku
    into a 9x9 numpy array with correct initial values. Staggers samples so that 10% of specified sample number have 
    0.05,0.1,0.15,0.2,0.25,0.30,0.35,0.4,0.45,0.5
    percentage blanks.'''

    # load image data
    font_images = get_digit_images()
    
    images, annotations = [], []
    
    for i in range(n):
        # create a sudoku grid = response
        grid_values, grid_fonts = get_sudoku_grid(font_images,
                                                  blank_percentage)
        
        # create the correspondong sudoku image array = input
        image, annotation = get_sudoku_image_and_annotation(grid_values,
                                        grid_fonts,
                                        font_images,
                                        as_array = False)
        
        # append image
        images.append(image)
        
        # add filename to annotation and append
        annotation['filename'] = 'image_' + str(i) + '.PNG'
        annotations.append(annotation)
        
    annotations = dict([(a['filename'],a) for a in annotations])
        
    # export data files
    data_dir = os.path.join(data_set_outdir,data_set_out_name)
    
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    for i,image in enumerate(images):
        # save image
        image_path = os.path.join(data_dir,'image_' + str(i) + '.PNG')
        image.save(image_path,"PNG")
        
    # save annotations json
    annotation_path = os.path.join(data_dir,'via_region_data.json')
    with open(annotation_path, 'w') as f:
        json.dump(annotations, f)
            
    return images, annotations

def main():
    # create data
    print('Started creating data @ ',datetime.now())
    n_train, n_val = 1000,200
    images, annotations = create_sudoku_data_set(n = n_train,
                                                 data_set_out_name = 'train')
    
    images, annotations = create_sudoku_data_set(n = n_val,
                                                 data_set_out_name = 'val')
    
    print('Finished creating data @ ',datetime.now(), '. Saving data ...')
        
if __name__ == '__main__':
    main()