# Import PuLP modeler functions
from pulp import *

# A list of strings from "1" to "9" is created
#Sequence = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
Index = [0,1,2,3,4,5,6,7,8]
Sequence = [1,2,3,4,5,6,7,8,9]

# The Vals, Rows and Cols sequences all follow this form
Vals = Sequence
Rows = Index
Cols = Index

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
                        
# The starting numbers are entered as constraints. Note: choices is indexed by value, row, column.                
# =============================================================================
# prob += choices["2"]["1"]["2"] == 1,""
# prob += choices["9"]["1"]["5"] == 1,""
# prob += choices["5"]["1"]["8"] == 1,""
# prob += choices["3"]["1"]["9"] == 1,""
# prob += choices["9"]["2"]["1"] == 1,""
# prob += choices["3"]["3"]["4"] == 1,""
# prob += choices["5"]["3"]["5"] == 1,""
# prob += choices["4"]["3"]["6"] == 1,""
# prob += choices["7"]["4"]["3"] == 1,""
# prob += choices["6"]["4"]["6"] == 1,""
# prob += choices["8"]["4"]["7"] == 1,""
# prob += choices["9"]["4"]["8"] == 1,""
# prob += choices["1"]["5"]["1"] == 1,""
# prob += choices["5"]["5"]["3"] == 1,""
# prob += choices["2"]["5"]["7"] == 1,""
# prob += choices["9"]["6"]["3"] == 1,""
# prob += choices["5"]["6"]["4"] == 1,""
# prob += choices["2"]["6"]["6"] == 1,""
# prob += choices["4"]["6"]["9"] == 1,""
# prob += choices["8"]["7"]["4"] == 1,""
# prob += choices["2"]["7"]["5"] == 1,""
# prob += choices["6"]["7"]["9"] == 1,""
# prob += choices["8"]["8"]["1"] == 1,""
# prob += choices["7"]["8"]["4"] == 1,""
# prob += choices["7"]["9"]["1"] == 1,""
# prob += choices["5"]["9"]["6"] == 1,""
# prob += choices["4"]["9"]["7"] == 1,""
# =============================================================================

prob += choices[1][0][2] == 1,""
prob += choices[9][0][4] == 1,""
prob += choices[5][0][7] == 1,""
prob += choices[3][0][8] == 1,""
prob += choices[9][1][0] == 1,""
prob += choices[3][2][3] == 1,""
prob += choices[5][2][4] == 1,""
prob += choices[4][2][5] == 1,""
prob += choices[7][3][2] == 1,""
prob += choices[6][3][5] == 1,""
prob += choices[8][3][6] == 1,""
prob += choices[9][3][7] == 1,""
prob += choices[1][4][0] == 1,""
prob += choices[5][4][2] == 1,""
prob += choices[2][4][6] == 1,""
prob += choices[9][5][2] == 1,""
prob += choices[5][5][3] == 1,""
prob += choices[2][5][5] == 1,""
prob += choices[4][5][8] == 1,""
prob += choices[8][6][3] == 1,""
prob += choices[2][6][4] == 1,""
prob += choices[6][6][8] == 1,""
prob += choices[8][7][0] == 1,""
prob += choices[7][7][3] == 1,""
prob += choices[7][8][0] == 1,""
prob += choices[5][8][5] == 1,""
prob += choices[4][8][6] == 1,""

# The problem data is written to an .lp file
prob.writeLP("Sudoku.lp")

# The problem is solved using PuLP's choice of Solver
prob.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

# A file called sudokuout.txt is created/overwritten for writing to
sudokuout = open('sudokuout.txt','w')

# The solution is written to the sudokuout.txt file 
for r in Rows:
    #if r == "1" or r == "4" or r == "7":
    if r == 0 or r == 3 or r == 6:
                    sudokuout.write("+-------+-------+-------+\n")
    for c in Cols:
        for v in Vals:
            if value(choices[v][r][c])==1:
                               
                #if c == "1" or c == "4" or c =="7":
                if c == 0 or c == 3 or c ==6:
                    sudokuout.write("| ")
                    
                sudokuout.write(str(v) + " ")
                
                #if c == "9":
                if c == 8:
                    sudokuout.write("|\n")
sudokuout.write("+-------+-------+-------+")                    
sudokuout.close()

# The location of the solution is give to the user
print("Solution Written to sudokuout.txt")

# --- test rest api
from sudoku_solver_api import InitialValueConstraint,InitialValueConstraints
import requests

constraints = [
    InitialValueConstraint(value=1,row_index=0,column_index=2), # prob += choices[1][0][2] == 1,""
    InitialValueConstraint(value=9,row_index=0,column_index=4), # prob += choices[9][0][4] == 1,""
    InitialValueConstraint(value=5,row_index=0,column_index=7), # prob += choices[5][0][7] == 1,""
    InitialValueConstraint(value=3,row_index=0,column_index=8), # prob += choices[3][0][8] == 1,""
    InitialValueConstraint(value=9,row_index=1,column_index=0), # prob += choices[9][1][0] == 1,""
    InitialValueConstraint(value=3,row_index=2,column_index=3), # prob += choices[3][2][3] == 1,""
    InitialValueConstraint(value=5,row_index=2,column_index=4), # prob += choices[5][2][4] == 1,""
    InitialValueConstraint(value=4,row_index=2,column_index=5), # prob += choices[4][2][5] == 1,""
    InitialValueConstraint(value=7,row_index=3,column_index=2), # prob += choices[7][3][2] == 1,""
    InitialValueConstraint(value=6,row_index=3,column_index=5), # prob += choices[6][3][5] == 1,""
    InitialValueConstraint(value=8,row_index=3,column_index=6), # prob += choices[8][3][6] == 1,""
    InitialValueConstraint(value=9,row_index=3,column_index=7), # prob += choices[9][3][7] == 1,""
    InitialValueConstraint(value=1,row_index=4,column_index=0), # prob += choices[1][4][0] == 1,""
    InitialValueConstraint(value=5,row_index=4,column_index=2), # prob += choices[5][4][2] == 1,""
    InitialValueConstraint(value=2,row_index=4,column_index=6), # prob += choices[2][4][6] == 1,""
    InitialValueConstraint(value=9,row_index=5,column_index=2), # prob += choices[9][5][2] == 1,""
    InitialValueConstraint(value=5,row_index=5,column_index=3), # prob += choices[5][5][3] == 1,""
    InitialValueConstraint(value=2,row_index=5,column_index=5), # prob += choices[2][5][5] == 1,""
    InitialValueConstraint(value=4,row_index=5,column_index=8), # prob += choices[4][5][8] == 1,""
    InitialValueConstraint(value=8,row_index=6,column_index=3), # prob += choices[8][6][3] == 1,""
    InitialValueConstraint(value=2,row_index=6,column_index=4), # prob += choices[2][6][4] == 1,""
    InitialValueConstraint(value=6,row_index=6,column_index=8), # prob += choices[6][6][8] == 1,""
    InitialValueConstraint(value=8,row_index=7,column_index=0), # prob += choices[8][7][0] == 1,""
    InitialValueConstraint(value=7,row_index=7,column_index=3), # prob += choices[7][7][3] == 1,""
    InitialValueConstraint(value=7,row_index=8,column_index=0), # prob += choices[7][8][0] == 1,""
    InitialValueConstraint(value=5,row_index=8,column_index=5), # prob += choices[5][8][5] == 1,""
    InitialValueConstraint(value=4,row_index=8,column_index=6)] # prob += choices[4][8][6] == 1]

initial_grid = InitialValueConstraints(initial_values = constraints).json()

sudoku_app_url = 'http://127.0.0.1:8000'

r1 = requests.get(sudoku_app_url + '/')
print(r1)

r2 = requests.post(sudoku_app_url + '/solve',data = initial_grid)
print(r2.text)


























    
    
    
    
    
    
    
    
    