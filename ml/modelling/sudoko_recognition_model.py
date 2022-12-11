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
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

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

_, width, height = X[0].shape
_, sudoku_width, sudoku_height = y[0].shape

# shape inputs - (n_data, 250, 250, 1)
X = np.concatenate(X,axis=0).reshape(-1,width,height,1) # image tensor needs to be 4-dimensional for tensorflow

# shape and one hot encode the character class targets - (n_data, 9, 9, n_classes = 10)
y = np.concatenate(y,axis=0).reshape(-1,sudoku_width,sudoku_height)
mapping = {'blank':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}

for i in range(y.shape[0]):
    for j in range(sudoku_width):
        for k in range(sudoku_height):
            y[i,j,k] = mapping[y[i,j,k]]
            
y = y.astype(int).reshape(-1)
temp = np.zeros((y.size, 10))
temp[np.arange(y.size),y] = 1
y = temp.reshape(-1,sudoku_width, sudoku_height, 10)

n_data = X.shape[0]
n_classes = 10

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
def get_weights(initializer = tf.initializers.glorot_uniform,
                channel_base = 3,
                kernel_sizes = [8,8,8],
                n_classes = 10):
    
    weights = {
    'wc1': tf.get_variable('W0', shape=(kernel_sizes[0],kernel_sizes[0],1,channel_base), initializer=initializer), 
    'wc2': tf.get_variable('W1', shape=(kernel_sizes[1],kernel_sizes[1],channel_base,2*channel_base), initializer=initializer), 
    'wc3': tf.get_variable('W2', shape=(kernel_sizes[2],kernel_sizes[2],2*channel_base,n_classes), initializer=initializer),
    'wd1': tf.get_variable('W3', shape=(810,810), initializer=initializer)
    }
    
    biases = {
        'bc1': tf.get_variable('B0', shape=(channel_base), initializer=initializer),
        'bc2': tf.get_variable('B1', shape=(2*channel_base), initializer=initializer),
        'bc3': tf.get_variable('B2', shape=(n_classes), initializer=initializer),
        'bd1': tf.get_variable('B3', shape=(810), initializer=initializer)
    }
    
    return weights, biases
    
def conv2d(x, W, b, strides=2,padding='SAME'):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
    
def maxpool2d(x, strides=2,padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, strides, strides, 1], strides=[1, strides, strides, 1],padding='SAME')

def conv_net(X, n_classes = 10):
    '''Builds tensorflow model graph and returns handles to relevant graph nodes in dictionary.'''
    
# =============================================================================
#     SAME
#     out_height = ceil(float(in_height) / float(strides[1]))
#     out_width  = ceil(float(in_width) / float(strides[2]))
# 
#     VALID
#     out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
#     out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))
#     
# =============================================================================
    weight, bias = get_weights(n_classes = n_classes)
    # (batchsize,250,250,1)
    conv1 = conv2d(X,weight['wc1'],bias['bc1'],strides = 2, padding = 'SAME')
    # (batchsize,125,125,3)
    max1 = maxpool2d(conv1, strides=2, padding = 'SAME')
    # (batchsize,63,63,3)
    conv2 = conv2d(max1, weight['wc2'], bias['bc2'],strides = 2, padding = 'SAME')
    # (batchsize,32,32,6)
    max2 = maxpool2d(conv2, strides=2, padding = 'SAME')
    # (batchsize,16,16,6)
    conv3 = conv2d(max2, weight['wc3'],bias['bc3'], strides = 1, padding = 'VALID')
    # (batchsize,9,9,10)
    fc1 = tf.reshape(conv3, [-1, weight['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weight['wd1']), bias['bd1'])
    fc1 = tf.nn.relu(fc1)
    # (batchsize, 810)
    out = tf.reshape(fc1,[-1, 9, 9, n_classes])
    # (batchsize, 9, 9, 10)
    
    return out

# --- build model graph
tf.reset_default_graph()

learning_rate = 0.1

X_place = tf.placeholder('float', [None, width, height, 1])
y_place = tf.placeholder('float', [None, 9,9,n_classes])

pred = conv_net(X_place)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels = y_place, 
        logits = pred)
    )

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_place, 1))

#calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --- train model
# Initializing the variables
training_iters = 5
batch_size = 300

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    valid_loss = []
    train_accuracy = []
    valid_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        
        batch_losses, batch_accuracy = [], []
        
        for batch in range(len(X)//batch_size):
            batch_x = X[batch*batch_size:min((batch+1)*batch_size,len(X))]
            batch_y = y[batch*batch_size:min((batch+1)*batch_size,len(y))]    
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={X_place: batch_x,
                                                 y_place: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={X_place: batch_x,
                                                              y_place: batch_y})
            
            batch_losses.append(loss)
            batch_accuracy.append(acc)
            
            print("Iter " + str(i) + ", Batch loss= " + \
                      "{:.6f}".format(loss) + ", Training batch Accuracy= " + \
                      "{:.5f}".format(acc))
        
        batch_losses, batch_accuracy = np.mean(batch_losses), np.mean(batch_accuracy)
        
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(batch_losses) + ", Training Accuracy= " + \
                      "{:.5f}".format(batch_accuracy))

        # Calculate accuracy for all 10000 mnist test images
        valid_acc,valid_l = sess.run([accuracy,cost], feed_dict={X_place: X_validate,y_place : y_validate})
        train_loss.append(batch_losses)
        valid_loss.append(valid_l)
        train_accuracy.append(batch_accuracy)
        valid_accuracy.append(valid_acc)
        print("Validation Accuracy:","{:.5f}".format(valid_acc))
    print("Optimization Finished!")
    summary_writer.close()