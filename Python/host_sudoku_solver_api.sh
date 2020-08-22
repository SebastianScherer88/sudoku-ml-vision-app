#!/bin/bash
uvicorn sudoku_solver_api:app --reload --host 0.0.0.0 --port 8000
