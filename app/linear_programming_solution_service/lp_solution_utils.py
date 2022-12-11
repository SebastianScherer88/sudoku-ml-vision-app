from pydantic import BaseModel, validator
from typing import List
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpInteger, value

INITIAL_CONSTRAINT_VALUE_RANGE = [1,2,3,4,5,6,7,8,9]
ROW_INDEX_RANGE = COLUMN_INDEX_RANGE = [0,1,2,3,4,5,6,7,8]

class InitialValueConstraint(BaseModel):
    
    row_index: int
    column_index: int
    value: int
    
    @validator('row_index')
    def check_row_range(cls, v):
        if v not in ROW_INDEX_RANGE:
            raise ValueError(f'Row index {v} is not in allowed row index range {ROW_INDEX_RANGE}')
        return v
    
    @validator('column_index')
    def check_column_range(cls, v):
        if v not in range(9):
            raise ValueError(f'Column index {v} is not in allowed column index range {COLUMN_INDEX_RANGE}')
        return v
    
    @validator('value')
    def check_initial_value_range(cls, v):
        if v not in INITIAL_CONSTRAINT_VALUE_RANGE:
            raise ValueError(f'Initital value {v} is not in allowed initial constraint value range {INITIAL_CONSTRAINT_VALUE_RANGE}')
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
                    if d not in INITIAL_CONSTRAINT_VALUE_RANGE:
                        raise ValueError(f'Solution entry {d} is not in the allowed solution entry range {INITIAL_CONSTRAINT_VALUE_RANGE}')
        else:
            pass
                        
        return v
    
def solve_sudoku(initial_value_constraints: InitialValueConstraints):
    
    # --- set up Sudoku problem  
    # The boxes list is created, with the row and column index of each square in each box    
    Boxes =[]
    for i in range(3):
        for j in range(3):
            Boxes += [[(ROW_INDEX_RANGE[3*i+k],COLUMN_INDEX_RANGE[3*j+l]) for k in range(3) for l in range(3)]]
    
    # The prob variable is created to contain the problem data        
    prob = LpProblem("Sudoku Problem",LpMinimize)
    
    # The problem variables are created
    choices = LpVariable.dicts("Choice",(INITIAL_CONSTRAINT_VALUE_RANGE,ROW_INDEX_RANGE,COLUMN_INDEX_RANGE),0,1,LpInteger)
    
    # The arbitrary objective function is added
    prob += 0, "Arbitrary Objective Function"
    
    # A constraint ensuring that only one value can be in each square is created
    for r in ROW_INDEX_RANGE:
        for c in COLUMN_INDEX_RANGE:
            prob += lpSum([choices[v][r][c] for v in INITIAL_CONSTRAINT_VALUE_RANGE]) == 1, ""
    
    # The row, column and box constraints are added for each value
    for v in INITIAL_CONSTRAINT_VALUE_RANGE:
        for r in ROW_INDEX_RANGE:
            prob += lpSum([choices[v][r][c] for c in COLUMN_INDEX_RANGE]) == 1,""
            
        for c in COLUMN_INDEX_RANGE:
            prob += lpSum([choices[v][r][c] for r in ROW_INDEX_RANGE]) == 1,""
    
        for b in Boxes:
            prob += lpSum([choices[v][r][c] for (r,c) in b]) == 1,""

    # add initial values if needed
    if initial_value_constraints != None and len(initial_value_constraints.initial_values) != 0:
        for initial_value_constraint in initial_value_constraints.initial_values:
            prob += choices[initial_value_constraint.value][initial_value_constraint.row_index][initial_value_constraint.column_index] == 1,""
            
    # --- solve sudoku with linear programing
    solution_status = prob.solve()

    if solution_status == -1:

        solved_grid = None
    elif solution_status == 1:

        solved_grid = [[None]*9 for i in range(9)]
        
        for row in ROW_INDEX_RANGE:
            for column in COLUMN_INDEX_RANGE:
                for val in INITIAL_CONSTRAINT_VALUE_RANGE:
                    if value(choices[val][row][column]) == 1:
                        solved_grid[row][column] = val
    
    return SudokuSolution(solved = solution_status,
                                solution = solved_grid)
