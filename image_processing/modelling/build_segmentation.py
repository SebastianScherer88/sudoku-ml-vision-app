from image_processing.settings import (
    LOCAL_TEMP_DIR
)

from app.parse_image_service.parse_image_utils import (
    preprocess_image,
    extract_grid_cell_patches
)

import cv2

sudoku_image = cv2.imread(initial_image.image_path)

preprocessed_sudoku_image = preprocess_image(sudoku_image)

_, _, image_patches, _ = extract_grid_cell_patches(preprocessed_sudoku_image)