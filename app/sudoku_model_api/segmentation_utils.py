# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 17:59:54 2020

@author: bettmensch
"""

import cv2
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from cell_recognizer_model import normalize_image

def get_bounding_rect(contour_array: np.array,
                      image_shape: tuple,
                      frame_width = 3):
    
    reduced_contour = contour_array[:,0,:]
    
    min_x, min_y = list(reduced_contour.min(0))
    max_x, max_y = list(reduced_contour.max(0))
    
    b1, b2, b3, b4 = min_y-frame_width, max_y+frame_width, min_x-frame_width, max_x+frame_width
    
    bounding_rect = np.zeros(image_shape)
    bounding_rect[b1:b2,b3:b4] = 1
    
    return [bounding_rect, (b1,b2,b3,b4)]

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

def extract_grid_cell_patches(sudoku_image: np.array,
                              resize = (500,500),
                              convert_to_gray_scale = True):
    
    # --- resize image
    sudoku_image = cv2.resize(sudoku_image, resize)
    
    # --- extract sudoku grid patch
    if convert_to_gray_scale:
        gray = cv2.cvtColor(sudoku_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = sudoku_image
        
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    
    for contour_i in contours:
        area = cv2.contourArea(contour_i)
        
        # find the outline of the entire grid
        if area > max(1000,max_area):
            max_area = area
            sudoku_grid_outline = contour_i
    
    sudoku_grid_mask = np.zeros((gray.shape),np.uint8)
    cv2.drawContours(sudoku_grid_mask,[sudoku_grid_outline],0,255,-1)
    cv2.drawContours(sudoku_grid_mask,[sudoku_grid_outline],0,0,2)
    
    sudoku_grid_patch = np.zeros_like(gray)
    sudoku_grid_patch[sudoku_grid_mask == 255] = gray[sudoku_grid_mask == 255]
    
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
            #cv2.imshow("sudoku_grid_mask: " + str(counter), sudoku_grid_mask)
            #cv2.waitKey()
            
        #print(counter)
    
    cell_patches, _ = cv2.findContours(sudoku_grid_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # get 81 biggest contour areas; they are the grid cell patches
    cell_patches_areas = [cv2.contourArea(cell_patch) for cell_patch in cell_patches]
    cell_patches_areas.sort()
    min_grid_cell_patches_area = cell_patches_areas[-81]
    
    counter_2 = 0
    sudoku_grid_mask_2 = np.zeros((gray.shape),np.uint8)
    cv2.drawContours(sudoku_grid_mask_2,[sudoku_grid_outline],0,255,-1)
    cv2.drawContours(sudoku_grid_mask_2,[sudoku_grid_outline],0,0,2)
    
    #cv2.imshow("sudoku_grid_mask 2 clean: " + str(counter_2), sudoku_grid_mask_2)
    #cv2.waitKey()
    
    cell_patches_final = []
    
    for i,cell_patch in enumerate(cell_patches):
        if cv2.contourArea(cell_patch) >= min_grid_cell_patches_area:
            cell_patches_final.append(cell_patch)
            counter_2 += 1
            cv2.drawContours(sudoku_grid_mask_2,cell_patches,i,(0, 255, 0), 3)
            #if counter_2 >= 54:
                #cv2.imshow("sudoku_grid_mask: " + str(counter_2), sudoku_grid_mask_2)
                #cv2.waitKey()
    
    cell_bounding_rectangles = [get_bounding_rect(cell_patch,sudoku_grid_mask_2.shape) for cell_patch in cell_patches_final]
    cell_bounding_rectangles = remove_overlapping_rectangles(cell_bounding_rectangles)
    
    mask_rect = np.zeros_like(sudoku_grid_mask)
    #image_patches_orig = []
    image_patches = []
    image_patch_coordinates = []
    
    for cell_rect_bundle in cell_bounding_rectangles:
        cell_rect, coords = cell_rect_bundle
        b1,b2,b3,b4 = coords
        mask_rect[cell_rect == 1] = sudoku_grid_patch[cell_rect == 1]
        #image_patches_orig.append(sudoku_grid_patch[b1:b2,b3:b4])
        image_patches.append(cv2.resize(sudoku_grid_patch[b1:b2,b3:b4],(100,100)))
        image_patch_coordinates.append(coords)
        
    return sudoku_image, sudoku_grid_mask, mask_rect, image_patches, image_patch_coordinates

def recognize_sudoku_grid(sudoku_image: np.array,
                          recognizer_model,
                          convert_to_gray_scale = True,
                          resize = (500,500)):
    ''' Takes an array representing an image of the sudoku grid with initial values and applies the digit recognitition pipeline by
    - extracting the grid cell patches
    - applying the digit recognition model
    and returning a 9x9 array of recognized initial value digits.'''
    
    sudoku_image, sudoku_grid_mask, mask_rect, image_patches, image_patch_coordinates = extract_grid_cell_patches(sudoku_image,
                                                                                                                  convert_to_gray_scale = convert_to_gray_scale,
                                                                                                                  resize = resize)
    
    image_patch_df = pd.DataFrame(data = image_patch_coordinates,
                                  columns = ['y_up','y_down','x_left','x_right'])
    
    formatted_image_patches = [normalize_image(np.expand_dims(image_patch,axis=[0,-1])) for image_patch in image_patches]
    
    # if image was parsed successfully into 81 grid cell patches, predict on those patches and assemble prediction grid
    if len(formatted_image_patches) == 81:
        image_patch_df['digit'] = [str(np.argmax(recognizer_model.predict(formatted_image_patch)) + 1) for formatted_image_patch in formatted_image_patches]
        
        # the input matrix field in the shiny dashboard expects string values 1-9, and an empty string to demark empty fields
        image_patch_df['digit'] = image_patch_df['digit'].replace({'10':''}).astype(str)
        
        image_patch_df = image_patch_df.sort_values(by = ['y_up'],
                                       ascending = True)
        
        image_patch_df['row_index'] = [i for i in range(1,10) for j in range(9)]
        
        image_patch_df = image_patch_df.sort_values(by = ['x_left'],
                                       ascending = True)
        
        image_patch_df['column_index'] = [i for i in range(1,10) for j in range(9)]
        
        parsed_grid = image_patch_df[['row_index','column_index','digit']]. \
            sort_values(['row_index','column_index'])['digit']. \
            values. \
            reshape(9,9)
            
        # convert to nested list for API ParsedSudokuImage response model class
        parsed_grid = [list(row) for row in parsed_grid]
            
        was_successful = True
    
    # if image was not parsed successfully, return error flag
    elif len(formatted_image_patches) != 81:
        
        # the input matrix field in the shiny dashboard expects string values 1-9, and an empty string to demark empty fields
        was_successful, parsed_grid = False, [[''] * 9 for i in range(9)]
    
    return was_successful, parsed_grid

# =============================================================================
# recognizer_model = load_model(r'C:/Users/bettmensch/GitReps/sudoku_solver/model/grid_cell_classifier')
# 
# #sudoku_image = cv2.imread(r"C:/Users/bettmensch/GitReps/Mask_RCNN/datasets/sudoku/val_original/sudoku_1.jpg")
# #sudoku_image = cv2.imread(r"C:/Users/bettmensch/GitReps/sudoku_solver/data/sudoku_recognition/snapshots/20200620_165814.jpg")
# #sudoku_image = cv2.imread(r"C:/Users/bettmensch/GitReps/sudoku_solver/data/sudoku_recognition/snapshots/20200620_165827.jpg")
# #sudoku_image = cv2.imread(r"C:/Users/bettmensch/GitReps/sudoku_solver/data/sudoku_recognition/snapshots/20200620_165824.jpg")
# sudoku_image = cv2.imread(r"C:/Users/bettmensch/GitReps/sudoku_solver/data/segmentation_validate/20200720_155952.jpg")
# sudoku_image = cv2.imread(r"C:/Users/bettmensch/GitReps/sudoku_solver/data/segmentation_validate/20200720_160015.jpg")
# sudoku_image = cv2.imread(r"C:/Users/bettmensch/GitReps/sudoku_solver/data/segmentation_validate/20200720_155950.jpg")
# sudoku_image = cv2.imread(r"C:/Users/bettmensch/GitReps/sudoku_solver/data/segmentation_validate/20200720_155954.jpg")
# sudoku_image = cv2.imread(r"C:/Users/bettmensch/GitReps/sudoku_solver/data/segmentation_validate/20200720_224345.jpg")
# sudoku_image = cv2.imread(r"C:/Users/bettmensch/GitReps/sudoku_solver/data/segmentation_validate/20200720_224750.jpg")
# sudoku_image = cv2.imread(r"C:/Users/bettmensch/GitReps/sudoku_solver/data/segmentation_validate/20200720_224756.jpg")
# 
# image, white_cell_patches, image_cell_patches, image_cell_patches_list, image_patch_coordinates = extract_grid_cell_patches(sudoku_image,
#                                                                                                                             resize = (int(0.563*1000),1000))
# 
# sudoku_grid = recognize_sudoku_grid(sudoku_image,
#                                    recognizer_model,
#                                    resize = (int(0.563*1000),1000))
# print(sudoku_grid)
# 
# cv2.imshow("mask_rect", image)
# cv2.waitKey()
# 
# cv2.imshow("mask_rect", white_cell_patches)
# cv2.waitKey()
# 
# cv2.imshow("mask_rect", image_cell_patches)
# cv2.waitKey()
# 
# 
# for patch in image_cell_patches_list[-20:]:
#    cv2.imshow("mask_rect", patch)
#    cv2.waitKey()
# =============================================================================

def main():
    snapshot_dir= r'C:/Users/bettmensch/GitReps/sudoku_solver/data/sudoku_recognition/snapshots'
    extraction_patch_dir = r'C:/Users/bettmensch/GitReps/sudoku_solver/data/sudoku_recognition/extracted_snapshot_digits'
    
    extracted_patch_counter = 0
    
    for image in os.listdir(snapshot_dir):
        snapshot_path = os.path.join(snapshot_dir,image)
        snapshot = cv2.imread(snapshot_path)
        
        _, _, _, extracted_cells, extracted_cell_positions = extract_grid_cell_patches(snapshot)
        
        for extracted_cell in extracted_cells:
            extracted_cell_path = os.path.join(extraction_patch_dir,'sudoku_cell_' + str(extracted_patch_counter) + '.png')
            cv2.imwrite(extracted_cell_path, extracted_cell)
            
            extracted_patch_counter += 1    

if __name__ == '__main__':
    main()