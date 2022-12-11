from fastapi import FastAPI
from app.linear_programming_solution_service.lp_solution_utils import (
    InitialValueConstraints,
    SudokuSolution,
    solve_sudoku
)

app = FastAPI()

@app.get('/')
def root():
    return {'message':'Hello World!'}

@app.post('/solve', response_model = SudokuSolution)
def apply_sudoku_solver(initial_value_constraints: InitialValueConstraints):
    
    solved_sudoku = solve_sudoku(initial_value_constraints)
    
    return solved_sudoku