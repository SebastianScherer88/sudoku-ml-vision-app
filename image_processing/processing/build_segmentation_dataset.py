from utils.s3 import download_directory
from image_processing.settings import (
    S3_DATA_BUCKET,
    S3_SUDOKU_GRID_PARSING_EXTRACT_DIR,
    LOCAL_TEMP_DIR
)

import os

download_directory(S3_DATA_BUCKET,S3_SUDOKU_GRID_PARSING_EXTRACT_DIR,os.path.join(LOCAL_TEMP_DIR,'sudoku-grid-images'))

