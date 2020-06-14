# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 17:59:54 2020

@author: bettmensch
"""

import cv2
import numpy as np

def get_bounding_rect(contour_array: np.array,
                      image_shape: tuple,
                      frame_width = 3):
    
    reduced_contour = contour_array[:,0,:]
    
    min_x, min_y = list(reduced_contour.min(0))
    max_x, max_y = list(reduced_contour.max(0))
    
    bounding_rect = np.zeros(image_shape)
    bounding_rect[(min_y-frame_width):(max_y+frame_width),(min_x-frame_width):(max_x+frame_width)] = 1
    
    return bounding_rect

def extract_grid_cell_patches(sudoku_image: np.array):
    
    # --- extract sudoku grid patch
    gray = cv2.cvtColor(sudoku_image, cv2.COLOR_BGR2GRAY)    
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
    thresh_patch = cv2.adaptiveThreshold(blur_patch, 255, 1, 1, 11, 2)    
    contours_patch, _ = cv2.findContours(thresh_patch, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for i,contour in enumerate(contours_patch):
        if cv2.contourArea(contour) > 400:
            cv2.drawContours(sudoku_grid_mask,contours_patch,i,(0, 255, 0), 7)
            
    cv2.imshow("mask", sudoku_grid_mask)
    cv2.waitKey()
    
    cell_patches, _ = cv2.findContours(sudoku_grid_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cell_bounding_rectangles = [get_bounding_rect(cell_patch,sudoku_grid_mask.shape) for cell_patch in cell_patches]
    
    mask_rect = np.zeros_like(sudoku_grid_mask)
    
    for cell_rect in cell_bounding_rectangles:
        mask_rect[cell_rect == 1] = sudoku_grid_patch[cell_rect == 1]
        
    return mask_rect
        
sudoku_image = cv2.imread(r"C:/Users/bettmensch/GitReps/Mask_RCNN/datasets/sudoku/val_original/sudoku_1.jpg")
sudoku_image = cv2.imread(r"C:/Users/bettmensch/GitReps/Mask_RCNN/datasets/sudoku/val_original/sudoku_2.png")
sudoku_image = cv2.imread(r"C:/Users/bettmensch/GitReps/Mask_RCNN/datasets/sudoku/val_original/sudoku_3.jpg")
sudoku_image = cv2.imread(r"C:/Users/bettmensch/GitReps/Mask_RCNN/datasets/sudoku/val_original/sudoku_4.jpg")
sudoku_image = cv2.imread(r"C:/Users/bettmensch/GitReps/Mask_RCNN/datasets/sudoku/val_original/sudoku_5.png")

mask_rect = extract_grid_cell_patches(sudoku_image)

cv2.imshow("mask_rect", mask_rect)
cv2.waitKey()