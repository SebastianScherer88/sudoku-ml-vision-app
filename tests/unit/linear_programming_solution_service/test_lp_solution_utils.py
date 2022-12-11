import pytest
from app.linear_programming_solution_service.lp_solution_utils import (
    InitialValueConstraint,
    InitialValueConstraints,
    SudokuSolution,
    solve_sudoku
)
    
@pytest.mark.parametrize(
    'input_constraints, expected_solution',
    [
        (InitialValueConstraints(initial_values=[
            InitialValueConstraint(row_index=0, 
                                   column_index=1, 
                                   value=8),
            InitialValueConstraint(row_index=0, 
                                   column_index=4, 
                                   value=9),
            InitialValueConstraint(row_index=1, 
                                   column_index=5, 
                                   value=4),
            InitialValueConstraint(row_index=2, 
                                   column_index=1, 
                                   value=7),
            InitialValueConstraint(row_index=2, 
                                   column_index=6, 
                                   value=1),
            InitialValueConstraint(row_index=2, 
                                   column_index=7, 
                                   value=8),
            InitialValueConstraint(row_index=2, 
                                   column_index=8, 
                                   value=5),
            InitialValueConstraint(row_index=3, 
                                   column_index=1, 
                                   value=3),
            InitialValueConstraint(row_index=3, 
                                   column_index=2, 
                                   value=2),
            InitialValueConstraint(row_index=3, 
                                   column_index=5, 
                                   value=8),
            InitialValueConstraint(row_index=4, 
                                   column_index=0, 
                                   value=1),
            InitialValueConstraint(row_index=4, 
                                   column_index=3, 
                                   value=6),
            InitialValueConstraint(row_index=4, 
                                   column_index=5, 
                                   value=2),
            InitialValueConstraint(row_index=4, 
                                   column_index=8, 
                                   value=4),
            InitialValueConstraint(row_index=5, 
                                   column_index=3, 
                                   value=9),
            InitialValueConstraint(row_index=5, 
                                   column_index=6, 
                                   value=7),
            InitialValueConstraint(row_index=5, 
                                   column_index=7, 
                                   value=1),
            InitialValueConstraint(row_index=6, 
                                   column_index=0, 
                                   value=6),
            InitialValueConstraint(row_index=6, 
                                   column_index=1, 
                                   value=2),
            InitialValueConstraint(row_index=6, 
                                   column_index=2, 
                                   value=3),
            InitialValueConstraint(row_index=6, 
                                   column_index=7, 
                                   value=4),
            InitialValueConstraint(row_index=7, 
                                   column_index=3, 
                                   value=5),
            InitialValueConstraint(row_index=8, 
                                   column_index=4, 
                                   value=2),
            InitialValueConstraint(row_index=8, 
                                   column_index=7, 
                                   value=6),
        ]
                                 ),
         SudokuSolution(
             solved=1,
             solution=[
                 [3, 8, 5, 1, 9, 7, 4, 2, 6],
                 [2, 1, 6, 8, 5, 4, 3, 9, 7],
                 [9, 7, 4, 2, 6, 3, 1, 8, 5],
                 [7, 3, 2, 4, 1, 8, 6, 5, 9],
                 [1, 5, 9, 6, 7, 2, 8, 3, 4],
                 [4, 6, 8, 9, 3, 5, 7, 1, 2],
                 [6, 2, 3, 7, 8, 9, 5, 4, 1],
                 [8, 9, 1, 5, 4, 6, 2, 7, 3],
                 [5, 4, 7, 3, 2, 1, 9, 6, 8]
             ]
         )
         ),

    ]
)
def test_solve_sudoku(input_constraints, expected_solution):
    
    actual_solution = solve_sudoku(input_constraints)
        
    assert actual_solution == expected_solution