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
import pickle

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



def get_digit_images(font = 'calibri',
                     base_path = r'C:\Users\bettmensch\GitReps\sudoku_solver\data\digits\individual_pics'):
    '''Helper function that loads in .png images of digits 1-9 of a specified font.'''
    
    assert font in ('calibri')
    assert os.path.isdir(base_path)

    digit_file_directory = os.path.join(base_path,font)

    digit_image_paths = [os.path.join(digit_file_directory,image_name) for image_name in os.listdir(digit_file_directory)]
    digit_names = [file_name.replace('.png','') for file_name in os.listdir(digit_file_directory)]
    digit_images = [Image.open(digit_image_path) for digit_image_path in digit_image_paths]
    
    images = dict([(k,v) for k,v in zip(digit_names, digit_images)])
    images['blank'] = Image.new('RGB', (digit_images[0].width, digit_images[0].height),(255,255,255))
    
    return images

def get_sudoku_grid(images: dict,
                    blank_ratio = 0.8):
    '''Helper function that returns the 81x81 integer array with integers.
    Serves as the response to the convolutional network.'''
    
    assert 0 <= blank_ratio and blank_ratio <= 1
    
    categories = list(images.keys())
    
    assert 'blank' in categories
    
    # get probability array
    probabilities = [(1 - blank_ratio) / (len(categories) - 1)]*len(categories)
    probabilities[categories.index('blank')] = blank_ratio
    
    grid_values = np.random.choice(a = categories,
                                   size = 81,
                                   replace = True,
                                   p = probabilities).reshape(9,9)
    
    return grid_values

def get_sudoku_image(sudoku_grid_array: np.array,
                     images: dict,
                     image_reduction = 0.463, # creates 250 x 250 images from original 540 x 540 size
                     as_array = True):
    '''Helper function that creates and returns an 540x540 image of a sudoku grid according to initial values
    contained in the sudoku_grid_array. Serves as the input to the convolutional network.'''
    
    assert sudoku_grid_array.shape == (9,9)
    
    # get individual dimensions
    width, height = images['blank'].width, images['blank'].height
    
    sudoku_image = Image.new('RGB',(width * 9, height * 9))
    
    # paste the digit images
    for row_index in range(9):
        for column_index in range(9):
            sudoku_image.paste(images[sudoku_grid_array[row_index,column_index]],
                              (column_index*width,row_index*height))
            
    # paste the vertical and horizontal dividing lines
    line_drawer = ImageDraw.Draw(sudoku_image)

    for row_index in range(10):
        line_drawer.line((0, row_index*height, width * 9, row_index*height), fill=(100, 100, 100), width=3)
    
    for column_index in range(10):
        line_drawer.line((column_index*width, 0, column_index*width, height * 9), fill=(100, 100, 100), width=3)
        
    # apply resizing
    width_resized, height_resized = int(image_reduction * sudoku_image.width), int(image_reduction * sudoku_image.height)
    
    sudoku_image = sudoku_image.resize((width_resized,height_resized))
        
    if as_array:
        # convert to 2-dim grey scale array representation
        sudoku_image = np.asarray(sudoku_image, dtype="int32" ).mean(axis = -1)
            
    return sudoku_image

def create_sudoku_data_set(n = 100000,
                           font = 'calibri',
                           base_path = r'C:\Users\bettmensch\GitReps\sudoku_solver\data\digits\individual_pics'):
    '''Helper function that creates inputs and response to the convolutional network meant to convert an image of a started sudoku
    into a 9x9 numpy array with correct initial values. Staggers samples so that 10% of specified sample number have 
    0.05,0.1,0.15,0.2,0.25,0.30,0.35,0.4,0.45,0.5
    percentage blanks.'''
    
    blank_percentages = list(map(lambda x: round(x,2),[0.8,0.5,0.1]))
    n_blank_class = int(n / len(blank_percentages))
    
    # load image data
    calibri_images = get_digit_images(font = font,
                                      base_path = base_path)
    
    grids = dict([(str(k),None) for k in blank_percentages])
    images = dict([(str(k),None) for k in blank_percentages])
    
    for blank_percentage in blank_percentages:
        
        print('Creating ', n_blank_class, ' sudoku grids/images with blank percentage ', blank_percentage)
        
        temp_grids, temp_images = [], []
        
        for i in range(n_blank_class):
            # create a sudoku grid = response
            sample_grid = get_sudoku_grid(calibri_images,
                                          blank_percentage)
            
            temp_grids.append(sample_grid)
            
            # create the correspondong sudoku image array = input
            sample_image = get_sudoku_image(sample_grid,
                                            calibri_images)
        
            temp_images.append(sample_image)
        
        # concatenate to higher dimensional arrays
        grids[str(blank_percentage)] = np.stack(temp_grids,axis=0) # of shape (n,9,9)
        images[str(blank_percentage)] = np.stack(temp_images,axis=0) # of shape (n,540,540,3)
    
    return images, grids

def main():
    # create data
    print('Started creating data @ ',datetime.now())
    n_data = 13500
    X, y = create_sudoku_data_set(n = n_data)
    
    print('Finished creating data @ ',datetime.now(), '. Saving data ...')
    
    # save data files
    for k in X.keys():
        
        X_k, y_k = X[k], y[k]
        n_data = X_k.shape[0]
        
        print('Saving blank percentage ', str(k), ' data files @ ',datetime.now())
        
        with open(r'./data/sudoku_recognition/sudoku_images_blank_percentage_' + str(k) + '_' + str(n_data) + '.pkl','wb') as X_file:
            pickle.dump(X_k,X_file)
            
        with open(r'./data/sudoku_recognition/sudoku_responses_blank_percentage_' + str(k) + '_' + str(n_data) + '.pkl','wb') as y_file:
            pickle.dump(y_k,y_file)
        
    print('Finished exporting data @ ',datetime.now())
        
if __name__ == '__main__':
    main()