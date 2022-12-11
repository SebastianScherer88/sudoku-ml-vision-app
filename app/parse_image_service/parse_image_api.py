import os
from fastapi import FastAPI
import cv2
from app.parse_image_service.parse_image_utils import (
    SudokuImage,
    ParsedSudokuImage,
    FailedToParseSudokuImage,
    preprocess_image,
    extract_grid_cell_patches
)

app = FastAPI()

@app.get('/')
def root():
    return {'message':'Hello World!'}
    
@app.post('/parse_image', response_model = ParsedSudokuImage)
def parse_image(initial_image: SudokuImage):
    
    sudoku_image = cv2.imread(initial_image.image_path)
    
    preprocessed_sudoku_image = preprocess_image(sudoku_image)
    
    _, _, image_patches, _ = extract_grid_cell_patches(preprocessed_sudoku_image)
    
    if len(image_patches) == 81:
        response = ParsedSudokuImage(image_parsed=True,
                                     cell_patches=image_patches)
    else:
        response = FailedToParseSudokuImage(image_parsed=False)
        
    return response