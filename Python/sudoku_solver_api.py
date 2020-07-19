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
                    
class InitialValueConstraint(BaseModel):
    
    row_index: int
    column_index: int
    value: int
    
    @validator('row_index')
    def check_row_range(cls, v):
        if v not in range(9):
            raise ValueError('Row index must be integer in [1,9]')
        return v
    
    @validator('column_index')
    def check_column_range(cls, v):
        if v not in range(9):
            raise ValueError('Column index must be integer in [1,9]')
        return v
    
    @validator('value')
    def check_initial_value_range(cls, v):
        if v not in range(1,10):
            raise ValueError('Initital value must be integer in [1,9]')
        return v
    
class InitialValueConstraints(BaseModel):
    
    initial_values: List[InitialValueConstraint] = []

class SudokuSolution(BaseModel):
    
    solved: int
    solution: List[List[int]] = None # outer lists are rows, inner lists are column indices
    
    @validator('solved')
    def check_solution_status(cls,v):
        if v not in (-1,1):
            raise ValueError('Sudoku solution status must be either 1 for solved, or -1 for failed.')
            
        return v
            
    @validator('solution')
    def check_solved_grid(cls,v):
        
        if v != None:
            if len(v) != 9:
                raise ValueError('Solution grid must have 9 rows.')
                
            for r in v:
                if len(r) != 9:
                    raise ValueError('Solution grid must have 9 columns.')
                    
                for d in r:
                    if d not in (1,2,3,4,5,6,7,8,9):
                        raise ValueError('Solution must contain integers in [1,9] only.')
        else:
            pass
                        
        return v
        
# =============================================================================
# 
# try:
#     constraint_1 = InitialValueConstraint(row_index = 2, column_index = 3, value = 6)
#     constraint_2 = InitialValueConstraint(row_index = 1, column_index = 7, value = 1)
#     constraint_3 = InitialValueConstraint(row_index = 5, column_index = 8, value = 3)
#     constraints = InitialValueConstraints(initial_values = [constraint_1, constraint_2, constraint_3])
#     for i in constraints.initial_values:
#         print(i.row_index)
#         
# except ValidationError as e:
#     print(e)
# 
# try:
#     solution = SudokuSolution(solved = 1,
#                               solution = [[1,2,3,4,5,6,7,8,9] for i in range(9)])
#     
#     print(solution)
# except ValidationError as e:
#     print(e)
# =============================================================================

@app.get('/')
def root():
    return {'message':'Hello World!'}
    
@app.post('/parse_image', response_model = ParsedSudokuImage)
def parse_image(initial_image: InitialImage):
    
    sudoku_image = cv2.imread(initial_image.image_path)

    was_parsing_successful, parsed_image = recognize_sudoku_grid(sudoku_image = sudoku_image,
                                                                 convert_to_gray_scale = True,
                                                                 recognizer_model = recognizer_model,
                                                                 resize = (IM_WIDTH, IM_HEIGHT))
    
    return ParsedSudokuImage(was_parsing_sucessful = was_parsing_successful,
                             parsed_values = parsed_image)

@app.post('/solve', response_model = SudokuSolution)
def solve_sudoku(initial_value_constraints: InitialValueConstraints):
    
    # --- set up Sudoku problem
    # The Vals, Rows and Cols sequences all follow this form
    Vals = [1,2,3,4,5,6,7,8,9]
    Rows = [0,1,2,3,4,5,6,7,8]
    Cols = [0,1,2,3,4,5,6,7,8]
    
    # The boxes list is created, with the row and column index of each square in each box
    Boxes =[]
    for i in range(3):
        for j in range(3):
            Boxes += [[(Rows[3*i+k],Cols[3*j+l]) for k in range(3) for l in range(3)]]
    
    # The prob variable is created to contain the problem data        
    prob = LpProblem("Sudoku Problem",LpMinimize)
    
    # The problem variables are created
    choices = LpVariable.dicts("Choice",(Vals,Rows,Cols),0,1,LpInteger)
    
    # The arbitrary objective function is added
    prob += 0, "Arbitrary Objective Function"
    
    # A constraint ensuring that only one value can be in each square is created
    for r in Rows:
        for c in Cols:
            prob += lpSum([choices[v][r][c] for v in Vals]) == 1, ""
    
    # The row, column and box constraints are added for each value
    for v in Vals:
        for r in Rows:
            prob += lpSum([choices[v][r][c] for c in Cols]) == 1,""
            
        for c in Cols:
            prob += lpSum([choices[v][r][c] for r in Rows]) == 1,""
    
        for b in Boxes:
            prob += lpSum([choices[v][r][c] for (r,c) in b]) == 1,""

    # add initial values if needed
    if initial_value_constraints != None and len(initial_value_constraints.initial_values) != 0:
        for initial_value_constraint in initial_value_constraints.initial_values:
            prob += choices[initial_value_constraint.value][initial_value_constraint.row_index][initial_value_constraint.column_index] == 1,""
            
    # --- solve sudoku with linear programing
    solution_status = prob.solve()

    if solution_status == -1:

        return SudokuSolution(solved = solution_status)
    elif solution_status == 1:

        solved_grid = [[None]*9 for i in range(9)]
        
        for row in Rows:
            for column in Cols:
                for val in Vals:
                    if value(choices[val][row][column]) == 1:
                        solved_grid[row][column] = val

        return SudokuSolution(solved = solution_status,
                              solution = solved_grid)
