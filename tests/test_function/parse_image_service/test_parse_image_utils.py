import pytest

from app.parse_image_service.parse_image_utils import (
    encode_image_file_for_http,
    decode_image_file_from_http,
    preprocess_image,
    extract_grid_cell_patches
)

@pytest.mark.parametrize(
    'image_path',
    [
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-1.PNG', # parsing failure
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-2.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-3.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-4.PNG', # parsing failure
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-5.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-6.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-7.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-8.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-9.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-10.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-11.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-12.PNG', # parsing failure
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-13.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-14.PNG', # parsing failure
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-15.PNG', # parsing failure
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-16.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-17.PNG', # parsing failure
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-18.PNG', # parsing failure
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-19.PNG', # parsing failure
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-20.PNG', # parsing failure
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-21.PNG', # parsing failure
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-22.PNG', # parsing failure
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-23.PNG', # parsing failure
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-24.PNG', # parsing failure
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-25.PNG', # parsing failure
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-26.PNG', # parsing failure
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-27.PNG',
    ]
)
def test_parse_image_pipeline(image_path):
    
    # load image & encode
    image_byte_string = encode_image_file_for_http(image_path)
    
    # decode image
    sudoku_image = decode_image_file_from_http(image_byte_string)
    
    # preprocess
    preprocessed_sudoku_image = preprocess_image(sudoku_image)
    
    # parse
    _, _, image_patches, _ = extract_grid_cell_patches(preprocessed_sudoku_image)
    
    assert len(image_patches) == 81
        
        
       