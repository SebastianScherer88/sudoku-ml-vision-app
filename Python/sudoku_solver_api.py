# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:38:07 2020

The sudoku solving restapi thats called from the rshiny app

@author: bettmensch
"""

from fastapi import FastAPI
from pydantic import BaseModel, validator, ValidationError
from typing import List
import numpy as np
from segmentation_utils import extract_grid_cell_patches
from cell_recognizer_model import normalize_image

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpInteger, value

app = FastAPI()

# --- body and response classes
IM_WIDTH, IM_HEIGHT = 500, 500

class InitialImage(BaseModel):

    image_array: List[List[float]] # flattened 2-dim grey scale image array; outer lists are rows, inner lists are column indices
    
    @validator('image_array')
    def check_solved_grid(cls,v):
        
        if v != None:
            if len(v) != IM_WIDTH:
                raise ValueError('Image width must be ' + str(IM_WIDTH) + ' pixels.')
                
            for r in v:
                if len(r) != IM_HEIGHT:
                    raise ValueError('Image height must be ' + str(IM_HEIGHT) + ' pixels.')
                    
        return v
        
class ParsedSudokuImage(BaseModel):

    parsed_values: List[List[int]] # outer lists are rows, inner lists are column indices

    @validator('parsed_values')
    def check_solved_grid(cls,v):
        
        if v != None:
            if len(v) != 9:
                raise ValueError('Solution grid must have 9 rows.')
                
            for r in v:
                if len(r) != 9:
                    raise ValueError('Solution grid must have 9 columns.')
                    
                for d in r:
                    if d not in (1,2,3,4,5,6,7,8,9,0):
                        raise ValueError('Solution must contain integers in [0,9] only (0 representing a blank).')
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
    
    image_array = np.arrage(initial_image.image_array)
    
    _, _, _, grid_cell_patches, extracted_cell_positions = extract_grid_cell_patches(image_array)
    
    grid_cell_patches_array = np.concatenate(grid_cell_patches,0)
    grid_cell_patch_digits = cell_recognition_model.predict(grid_cell_patches_array)
    
    recognized_grid_cells = []
    
    for cell_patch, cell_patch_position in zip(grid_cell_patches,extracted_cell_positions):
        x_left, x_right, y_top, y_bottom = cell_patch_position
        normalized_cell_patch = normalize_image(cell_patch)

    return

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
