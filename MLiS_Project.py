# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:32:40 2021

@author: OmarL
"""

import os
import numpy as np
import tensorflow as tf
import cv2 as cv
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pk

# Reading in Training Images + Minor Pre-processing
def read_train_imgs(colour, filename, resize):
    path = "training_data//training_data//"
    X_train = []
    
    for ii in os.listdir(path):
        # Avoids reading bad images. 
        if ii in ["10171.png", "3141.png", "4895.png", "8285.png", "3999.png"]:
            continue
        
        img = cv.imread(path + ii)
        if colour != None:
            img = cv.cvtColor(img, colour)
            
        # by default, images are (320 x 240). Resized down to (80 x 60)...
        # helps with memory demand
        img = cv.resize(img, resize) 
        X_train.append([int(''.join(filter(str.isdigit, ii))), img])
     
    
    X_train = sorted(X_train, key=lambda x: (x[0]))
    # remove column used for sorting
    X_train = [r.pop(1) for r in X_train]
    pk.dump(X_train, open(filename, 'wb'))
    
def read_test_imgs(colour, filename, resize):
    path = "test_data//test_data//"
    X_test = []
    
    for ii in os.listdir(path):
        
        img = cv.imread(path + ii)
        if colour != None:
            img = cv.cvtColor(img, colour)
            
        img = cv.resize(img, resize) 
        X_test.append(img)

    pk.dump(X_test, open(filename, 'wb'))

    
#cv.COLOR_RGB2GRAY
#cv.COLOR_RGB2HSV
#read_test_imgs(cv.COLOR_RGB2HSV, "X_test_HSV.pkl", (80, 60))
#read_train_imgs(None, "X_train_RGB.pkl", (80, 60))

# Reading in Training labels                                                
X_train = np.asarray(pk.load(open("X_train_Channels\X_train_GRAY.pkl", "rb")))
X_test = np.asarray(pk.load(open("X_test_Channels\X_test_GRAY.pkl", "rb")))

# Display images to check sorting has been done correctly
#cv.imshow('Window', X_train[1])
#cv.waitKey(1)

y_train = np.genfromtxt("training_norm.csv", delimiter=",")
y_train = y_train[1:,:]
#print(X_train.shape, y_train.shape)

# Norm data; scaled between 0 and 1.
X_train, X_test = X_train / 255., X_test / 255.




