from fastapi import FastAPI
import uvicorn
import cv2
from app.parse_image_service.parse_image_utils import (
    ParseRequest,
    ParsedImage,
    ParseResponse,
    decode_image_file_from_http,
    preprocess_image,
    extract_grid_cell_patches,
    encode_images_for_http
)

app = FastAPI()

@app.get('/')
def root():
    return {"status": "alive"}
    
@app.post('/parse_image', response_model = ParseResponse)
def parse_image(request: ParseRequest):
    
    parsed_images = []
    
    for image_byte_string in request.instances:

        sudoku_image = decode_image_file_from_http(image_byte_string)
        
        preprocessed_sudoku_image = preprocess_image(sudoku_image)
        
        _, _, image_patches, _ = extract_grid_cell_patches(preprocessed_sudoku_image)
        
        if len(image_patches) == 81:

            encoded_image_patches = encode_images_for_http(image_patches)
        
            image_parsed, parsed_cells = True, encoded_image_patches
        else:
            image_parsed, parsed_cells = False, ['',] * 81
            
        parsed_image = ParsedImage(image_parsed=image_parsed,
                                   parsed_cells = parsed_cells)
            
        parsed_images.append(parsed_image)
                
    return ParseResponse(instances=parsed_images)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)