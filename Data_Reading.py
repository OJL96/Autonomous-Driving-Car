# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:32:40 2021

@author: OmarL


NOTES: File reads data, resizes images, converts images to RGB,
       and saves results as a .pkl file. 

"""

import os
import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import pickle as pk

np.random.seed(123)
np.set_printoptions(suppress=True)

# Void function - reading in data
def read_data_imgs(filename, resize, colour):
    path = "training_data//turn//"
    data = []
    for ii in os.listdir(path):
        # Avoid reading corrupt images. 
        if ii in ["10171.png", "3141.png", "4895.png", "8285.png", "3999.png"]:
            continue
        img = cv.imread(path + ii)
        img = cv.cvtColor(img, colour)
            
        # by default, images are (320 x 240). Resized down to (80 x 60)...
        # helps with memory demand
        img = cv.resize(img, resize) 
        data.append([int(''.join(filter(str.isdigit, ii))), img])
     
    
    data = sorted(data, key=lambda x: (x[0]))
    # remove column used for sorting
    data = [r.pop(1) for r in data]

    pk.dump(data, open(filename, 'wb'))
    
# Void function - reading in holdout set
def read_holdout_imgs(filename, colour, resize):
    path = "test_data//test_data//"
    holdout = []
    
    for ii in os.listdir(path):
        
        img = cv.imread(path + ii)
        if colour != None:
            img = cv.cvtColor(img, colour)
            
        if resize != None:
            img = cv.resize(img, resize) 
        holdout.append([int(''.join(filter(str.isdigit, ii))), img])

    holdout = sorted(holdout, key=lambda x: (x[0]))
    # remove column used for sorting
    holdout = [r.pop(1) for r in holdout]
    pk.dump(holdout, open(filename, 'wb'))
    
# Manual Shuffle function
def shuffle_sets(X, y):
    
    concat_data = list(zip(X, y))
    np.random.shuffle(concat_data)
    X, y = list(zip(*concat_data))
    
    return np.asarray(X), np.asarray(y)

read_holdout_imgs( "Holdout_RGB_Resized.pkl", (80,60), cv.COLOR_BGR2RGB)
read_data_imgs("Data_RGB_Not_Resized_turn.pkl", (80,60), cv.COLOR_BGR2RGB)

# display image: used for testing
#cv.imshow('Window', cropped_image)
#cv.waitKey(0)
#cv.destroyAllWindows
















