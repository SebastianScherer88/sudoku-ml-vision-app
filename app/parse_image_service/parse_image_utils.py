import base64
from typing import List, Tuple, Union

import cv2
import numpy as np
from pydantic import BaseModel, validator
from pathlib import Path

import pandas as pd


class ParseRequest(BaseModel):

    instances: List[str] # strings are utf encodings of base64 encodings of image file bytes

class ParsedImage(BaseModel):
    
    image_parsed: bool = False
    parsed_cells: List[str] = ['',] * 81
    
    @validator('parsed_cells')
    def count_cell_patches(cls,parsed_cells):
        if len(parsed_cells) != 81:
            raise ValueError(f'The number of parsed cells for a successfully parsed Sudoku image must be 81, not {len(parsed_cells)}')
        
        return parsed_cells
    
class ParseResponse(BaseModel):
    
    instances: List[ParsedImage]
    

def decode_image_file_from_http(image_file_data: str):

    # undo base64 encoding step
    im_b = base64.b64decode(image_file_data)
        
    image_arr = np.frombuffer(im_b, np.uint8)
    image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR) # reads BGR color channel ordering (like the cv2.imread function - see above)
        
    return image

def encode_image_file_for_http(image_file_path: Union[Path,str]) -> str:
    '''
    Takes a file from disk and applies encoding convention to allow image data to be
    serialized to json.
    Useful for sending image data via HTTP.
    '''
    
    with open(image_file_path, 'rb') as open_file:
        im_bytes = open_file.read()
    
    # base64 encoding
    im_b64 = base64.b64encode(im_bytes)
    
    # utf decode so json can serialize it
    im_b64_str = im_b64.decode("utf8")
    
    return im_b64_str

def encode_images_for_http(images: List[np.array]) -> List[str]:
    
    encoded_images = []
    
    for i, image in enumerate(images):
        temp_image_path = './temp_image_patch_{i}.png'
        cv2.imwrite(temp_image_path,image)
        encoded_image = encode_image_file_for_http(temp_image_path)
        
        encoded_images.append(encoded_image)
        
    return encoded_images

def get_bounding_rect(contour_array: np.array,
                      image_shape: tuple,
                      frame_width = 3) -> Tuple[np.array, Tuple[int,int,int,int]]:
    
    reduced_contour = contour_array[:,0,:]
    
    min_x, min_y = list(reduced_contour.min(0))
    max_x, max_y = list(reduced_contour.max(0))
    
    b1, b2, b3, b4 = min_y-frame_width, max_y+frame_width, min_x-frame_width, max_x+frame_width
    
    bounding_rect = np.zeros(image_shape)
    bounding_rect[b1:b2,b3:b4] = 1
    
    return bounding_rect, (b1,b2,b3,b4)

def remove_overlapping_rectangles(bounding_rectangles: list,
                                  min_dim_index = 20):
    
    # get average width and height from largest 20 rectangles
    widths  = [rect[1][1] - rect[1][0] for rect in bounding_rectangles]
    heights  = [rect[1][3] - rect[1][2] for rect in bounding_rectangles]
    
    widths.sort()
    heights.sort()
    
    min_width, min_height = widths[-min_dim_index]/2, heights[-min_dim_index]/2
    min_overlap_area = min_width * min_height
    
    # assemble list of non-overlapping rectangles iteratively
    non_overlapping_rectangles = []
    
    for rectangle in bounding_rectangles:
        overlapping = False
        
        for vetted_rectangle in non_overlapping_rectangles:
            overlap_area = np.sum((rectangle[0] + vetted_rectangle[0]) == 2)/2
            
            if overlap_area > min_overlap_area:
                overlapping = True
                break
                
        if not overlapping:
            non_overlapping_rectangles.append(rectangle)
    
    return non_overlapping_rectangles

def preprocess_image(sudoku_image: np.array,
                     resize: Tuple[int,int] = None,
                     convert_to_grayscale: bool = True) -> np.array:
    
    # --- resize image
    if resize is None:
        resize = int(sudoku_image.shape[0]/sudoku_image.shape[1] * 500), 500
        
    resized_sudoku_image = cv2.resize(sudoku_image, resize)
    
    # --- extract sudoku grid patch
    if convert_to_grayscale:
        resized_sudoku_image = cv2.cvtColor(resized_sudoku_image, cv2.COLOR_BGR2GRAY)
        
    return resized_sudoku_image

def extract_grid_cell_patches(sudoku_image: np.array,
                              order_patches: bool = True):
        
    blur = cv2.GaussianBlur(sudoku_image, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    
    for contour_i in contours:
        area = cv2.contourArea(contour_i)
        
        # find the outline of the entire grid
        if area > max(1000,max_area):
            max_area = area
            sudoku_grid_outline = contour_i
    
    sudoku_grid_mask = np.zeros((sudoku_image.shape),np.uint8)
    cv2.drawContours(sudoku_grid_mask,[sudoku_grid_outline],0,255,-1)
    cv2.drawContours(sudoku_grid_mask,[sudoku_grid_outline],0,0,2)
    
    sudoku_grid_patch = np.zeros_like(sudoku_image)
    sudoku_grid_patch[sudoku_grid_mask == 255] = sudoku_image[sudoku_grid_mask == 255]
    
    # --- extract individual grid cell patches as rectanges
    blur_patch = cv2.GaussianBlur(sudoku_grid_patch, (5,5), 0)    
    thresh_patch = cv2.adaptiveThreshold(blur_patch, 255, 1, 1, 11, 3)    
    contour_patches, _ = cv2.findContours(thresh_patch, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # get 81st biggest patch area size; it will be the smalled grid cell patch we want to consider
    contour_patches_areas = [cv2.contourArea(contour_patch) for contour_patch in contour_patches]
    contour_patches_areas.sort()
    min_contour_patches_area = contour_patches_areas[-81]
    
    counter_1 = 0
    
    for i,contour_patch in enumerate(contour_patches):
        if cv2.contourArea(contour_patch) >= min_contour_patches_area:
            counter_1 += 1
            cv2.drawContours(sudoku_grid_mask,contour_patches,i,(0, 255, 0), 3)

    cell_patches, _ = cv2.findContours(sudoku_grid_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # get 81 biggest contour areas; they are the grid cell patches
    cell_patches_areas = [cv2.contourArea(cell_patch) for cell_patch in cell_patches]
    cell_patches_areas.sort()
    min_grid_cell_patches_area = cell_patches_areas[-81]
    
    counter_2 = 0
    sudoku_grid_mask_2 = np.zeros((sudoku_image.shape),np.uint8)
    cv2.drawContours(sudoku_grid_mask_2,[sudoku_grid_outline],0,255,-1)
    cv2.drawContours(sudoku_grid_mask_2,[sudoku_grid_outline],0,0,2)

    
    cell_patches_final = []
    
    for i,cell_patch in enumerate(cell_patches):
        if cv2.contourArea(cell_patch) >= min_grid_cell_patches_area:
            cell_patches_final.append(cell_patch)
            counter_2 += 1
            cv2.drawContours(sudoku_grid_mask_2,cell_patches,i,(0, 255, 0), 3)

    
    cell_bounding_rectangles = [get_bounding_rect(cell_patch,sudoku_grid_mask_2.shape) for cell_patch in cell_patches_final]
    cell_bounding_rectangles = remove_overlapping_rectangles(cell_bounding_rectangles)
    
    mask_rect = np.zeros_like(sudoku_grid_mask)
    image_patches = []
    image_patch_coordinates = []
    
    for cell_rect_bundle in cell_bounding_rectangles:
        cell_rect, coords = cell_rect_bundle
        b1,b2,b3,b4 = coords
        mask_rect[cell_rect == 1] = sudoku_grid_patch[cell_rect == 1]
        image_patches.append(cv2.resize(sudoku_grid_patch[b1:b2,b3:b4],(100,100)))
        image_patch_coordinates.append(coords)
        
    if order_patches:
        # assemble dataframe of image (meta) data
        image_patch_df = pd.DataFrame(data = image_patch_coordinates,
                                  columns = ['y_up','y_down','x_left','x_right'])
        image_patch_df['image_patches'] = image_patches
        image_patch_df['image_patch_coordinates'] = image_patch_coordinates
        
        # order by y coordinate, then by x coordinate
        image_patch_df = image_patch_df.sort_values(by = ['y_up'],ascending = True)
        image_patch_df['row_index'] = [i for i in range(9) for j in range(9)]
        
        image_patch_df = image_patch_df.sort_values(by = ['x_left'],ascending = True)
        image_patch_df['column_index'] = [i for i in range(9) for j in range(9)]
        
        image_patch_df = image_patch_df.sort_values(['row_index','column_index'])
        
        # extract ordered image (meta) data
        image_patches = image_patch_df['image_patches'].tolist()
        image_patch_coordinates = image_patch_df['image_patch_coordinates'].tolist()
        
    return sudoku_grid_mask, mask_rect, image_patches, image_patch_coordinates