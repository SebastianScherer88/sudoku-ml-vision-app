# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:38:07 2020

The sudoku solving restapi thats called from the rshiny app

@author: bettmensch
"""

import os
from fastapi import FastAPI
from pydantic import BaseModel, validator
from typing import List
import cv2
from tensorflow.keras.models import load_model
from segmentation_utils import recognize_sudoku_grid

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpInteger, value

app = FastAPI()

# load digit recognizer model
recognizer_model = load_model(r'../model/grid_cell_classifier')

# --- body and response classes
IM_WIDTH, IM_HEIGHT = 500, 500

class InitialImage(BaseModel):

    #image_array: List[List[float]] # flattened 2-dim grey scale image array; outer lists are rows, inner lists are column indices
    image_path: str
    
    @validator('image_path')
    def check_file_exists(cls,v):
        assert os.path.exists(v)
        
        return v
# =============================================================================
#     @validator('image_array')
#     def check_solved_grid(cls,v):
#         
#         if v != None:
#             if len(v) != IM_WIDTH:
#                 raise ValueError('Image width must be ' + str(IM_WIDTH) + ' pixels.')
#                 
#             for r in v:
#                 if len(r) != IM_HEIGHT:
#                     raise ValueError('Image height must be ' + str(IM_HEIGHT) + ' pixels.')
#                     
#         return v
# =============================================================================
        
class ParsedSudokuImage(BaseModel):

    was_parsing_sucessful: bool # whether the parsing of image for initial value extraction was successful
    parsed_values: List[List[str]] # outer lists are rows, inner lists are column indices

    @validator('parsed_values')
    def check_solved_grid(cls,v):
        
        if v != None:
            if len(v) != 9:
                raise ValueError('Solution grid must have 9 rows.')
                
            for r in v:
                if len(r) != 9:
                    raise ValueError('Solution grid must have 9 columns.')
                    
                for d in r:
                    if d not in [str(digit) for digit in range(1,10)] + ['',]:
                        raise ValueError('Solution must contain strings representing digits 1-9, or empty strings representing blanks.')
        else:
            pass
                        
        return v

@app.get('/')
def root():
    return {'message':'Hello World!'}
    
@app.post('/parse_image', response_model = ParsedSudokuImage)
def parse_image(initial_image: InitialImage):
    
    sudoku_image = cv2.imread(initial_image.image_path)
    
    re_size = int(sudoku_image.shape[1]/sudoku_image.shape[0] * 1000), 1000
    
    print(re_size)

    was_parsing_successful, parsed_image = recognize_sudoku_grid(sudoku_image = sudoku_image,
                                                                 convert_to_gray_scale = True,
                                                                 recognizer_model = recognizer_model,
                                                                 resize = re_size)
    
    return ParsedSudokuImage(was_parsing_sucessful = was_parsing_successful,
                             parsed_values = parsed_image)
