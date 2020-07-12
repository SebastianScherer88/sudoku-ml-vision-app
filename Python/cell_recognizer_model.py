# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 20:48:51 2020

@author: bettmensch
"""

import os
import cv2
import numpy as np

from keras.losses import categorical_crossentropy as xentropy
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.optimizers import Adam


def rotate_image(image, angle):
    
  average_gray = np.mean(image)
    
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  
  result[result == 0] = average_gray
  
  return result

def normalize_image(image):
    
    image = (image - np.mean(image)) / np.std(image)
    
    return image

# --- collect image data
image_dir = r'C:\Users\bettmensch\GitReps\sudoku_solver\data\sudoku_recognition\extracted_snapshot_digits'
digit_classes = os.listdir(image_dir)

n_class_size = 500
n_train = int(10 * 0.85 * n_class_size)

images = []
responses = []

for i,digit_class in enumerate(digit_classes):
    digit_dir = os.path.join(image_dir,digit_class)
    digit_images = [cv2.imread(os.path.join(digit_dir,image_name),cv2.IMREAD_GRAYSCALE) for image_name in os.listdir(digit_dir)[:n_class_size]]
    n_real_size = len(digit_images)

    rotated_digit_images = []

    for digit_image in digit_images:
        rotation_angle = np.random.choice([i for i in range(-15,15,1)])
        rotated_image = np.expand_dims(rotate_image(digit_image,rotation_angle),-1)
        
        normalized_image = normalize_image(rotated_image)
        
        rotated_digit_images.append(normalized_image)

    images.append(np.stack(rotated_digit_images,0))
    responses.extend([i] * n_real_size)
    
X = np.concatenate(images,0)
y = to_categorical(responses)

# shuffle
shuffled_index = np.random.permutation(X.shape[0])

X = X[shuffled_index]
y = y[shuffled_index]

image_no = 1111
cv2.imshow('ss',X[image_no,:,:])
cv2.waitKey()

y[image_no,:]

# --- build model
cell_shape = (100,100,1)
model = Sequential()
#add model layers
model.add(Conv2D(10, (3, 3), activation='relu', input_shape=cell_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(20, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(40, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation = 'softmax'))

print(model.summary())

opt = Adam(lr=0.01)

model.compile(optimizer= opt,
              loss=xentropy,
              metrics=['accuracy'])

history = model.fit(X[:n_train,:,:],
                    y[:n_train,:], 
                    epochs=4, 
                    batch_size = 200,
                    validation_data=(X[n_train:,:,:], y[n_train:,:]))

# save model
model.save(r'C:/Users/bettmensch/GitReps/sudoku_solver/model/grid_cell_classifier')