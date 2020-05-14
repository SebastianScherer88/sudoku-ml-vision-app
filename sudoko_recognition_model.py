# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:30:31 2020

This script trains a sudoku image recognition model using the 
artificial image data set created by the "image_processing.py" script
and a convolutional neural net built with Tensorflow.

@author: bettmensch
"""

import os
import pickle
import numpy as np
import gc
import tensorflow as tf

# --- data prep
image_data_directory = r'C:\Users\bettmensch\GitReps\sudoku_solver\data\sudoku_recognition'
image_data_files = map(lambda x: os.path.join(image_data_directory,x),os.listdir(image_data_directory))

image_data_files = [file for file in image_data_files if '0.8' in file]

# load response arrays
X, y = [], []

for image_data_file in image_data_files:
    image_data_file_name = os.path.split(image_data_file)[-1]
    
    with open(image_data_file,'rb') as file:
        image_data = pickle.load(file)
    
    if 'responses' in image_data_file_name:
        y.append(image_data)
    elif 'images' in image_data_file_name:
        X.append(image_data)
        
n_data, width, height = X.shape
n_data, sudoku_width, sudoku_height = y.shape
n_classes = 9
        
X = np.concatenate(X,axis=0).reshape(n_data,width,height,1) # image tensor needs to be 4-dimensional for tensorflow
y = np.concatenate(y,axis=0).reshape(n_data,sudoku_width * sudoku_height)

gc.collect()

# train validate test split
train, validate, test = int(0.8*n_data),int(0.1*n_data),int(0.1*n_data)
X, X_validate, X_test = X[:train,:,:,:], X[train:train + validate,:,:,:], X[train + validate:,:,:,:]
y, y_validate, y_test = y[:train,:,:], y[train:train + validate,:,:], y[train + validate:,:,:]

# centre inputs around 0 and scale
x_mean, x_var = X.mean(axis = 0), X.var(axis=0)
x_var[x_var == 0] = 1

X, X_validate, X_test = map(lambda X: X - x_mean / x_var, [X, X_validate, X_test])

# --- modelling
def get_weights(initializer = tf.contrib.layers.xavier_initializer,
                channel_base = 10,
                n_classes = 81):
    
    weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,channel_base), initializer=initializer), 
    'wc2': tf.get_variable('W1', shape=(3,3,channel_base,2*channel_base), initializer=initializer), 
    'wc3': tf.get_variable('W2', shape=(3,3,2*channel_base,3*channel_base), initializer=initializer), 
    'wd1': tf.get_variable('W3', shape=(32*3*channel_base,3*channel_base), initializer=initializer), # 250 x 250 -> 32 x 32 after 3 maxpools of 2x2 with stride 2
    'out': tf.get_variable('W6', shape=(128,n_classes), initializer=initializer), 
    }
    
    biases = {
        'bc1': tf.get_variable('B0', shape=(channel_base), initializer=initializer),
        'bc2': tf.get_variable('B1', shape=(2*channel_base), initializer=initializer),
        'bc3': tf.get_variable('B2', shape=(3*channel_base), initializer=initializer),
        'bd1': tf.get_variable('B3', shape=(128), initializer=initializer),
        'out': tf.get_variable('B4', shape=(n_classes), initializer=initializer),
    }
    
    return weights, biases
    
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
    
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def build_graph(X, n_classes = 81):
    '''Builds tensorflow model graph and returns handles to relevant graph nodes in dictionary.'''
    
    weight, bias = get_weights()
    
    conv1 = conv2d(X,weight['wc1'],bias['bc1'])
    
    max1 = maxpool2d(conv1)
    
    conv2 = conv2d(max1, weight['wc2'], bias['bc2'])
    
    max2 = maxpool2d(conv2)
    
    conv3 = conv2d(max2, weight['wc3'],bias['bc3'])
    
    fc1 = tf.reshape(conv3 [-1, weight['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weight['wd1']), bias['bd1'])
    fc1 = tf.nn.relu(fc1)
    
    out = tf.add(tf.matmul(fc1, weight['out']), bias['out'])
    
    return out

# --- build model
X = tf.placeholder('float', [None, width, height, 1])
y = tf.placeholder('float', [None, n_classes])

sudoku_grid_pred = build_graph(X)

