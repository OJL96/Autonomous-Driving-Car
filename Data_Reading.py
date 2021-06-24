# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:32:40 2021

@author: OmarL


NOTES: None

"""

import os
import numpy as np
np.random.seed(123)
np.set_printoptions(suppress=True)

import tensorflow as tf

import cv2 as cv
import matplotlib.pyplot as plt
import pickle as pk


# Void function - reading in data
def read_data_imgs( filename, resize):
    path = "training_data//turn//"
    data = []
    for ii in os.listdir(path):
        # Avoids reading bad images. 
        if ii in ["10171.png", "3141.png", "4895.png", "8285.png", "3999.png"]:
            continue
        
        img = cv.imread(path + ii)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
        # by default, images are (320 x 240). Resized down to (80 x 60)...
        # helps with memory demand
        #img = cv.resize(img, resize) 
        #data.append([int(''.join(filter(str.isdigit, ii))), img])
        data.append(img)
     
    
    #data = sorted(data, key=lambda x: (x[0]))
    # remove column used for sorting
    #data = [r.pop(1) for r in data]

    #pk.dump(data, open(filename, 'wb'))
    np.save("Data_RGB_Not_Resized_turn.npy", data)
    

# Void function - reading in holdout set
def read_holdout_imgs(colour, filename, resize):
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
    #pk.dump(holdout, open(filename, 'wb'))
    return holdout
    

def shuffle_sets(X, y):
    """
        Shuffle function
        
    """
    concat_data = list(zip(X, y))
    np.random.shuffle(concat_data)
    X, y = list(zip(*concat_data))
    
    return np.asarray(X), np.asarray(y)
    
def train_test_split(X, y, split=0.8):
    """
        Data splitting function 
        
    """
    X, y = shuffle_sets(X, y)
    X_train, y_train = X[:round(X.shape[0]*split)], y[:round(X.shape[0]*split)]
    X_test, y_test = X[X_train.shape[0]:], y[y_train.shape[0]:]
    
    return X_train, y_train, X_test, y_test


#holdout = np.asarray(read_holdout_imgs(None, None, None))
#read_data_imgs("Data_RGB_Not_Resized_turn.pkl", None)


# Reading in Training features                                                
X = np.asarray(pk.load(open("Data_Channels\Holdout_RGB_Not_Resized.pkl", "rb")))
#holdout= np.asarray(pk.load(open("Holdout_Channels\Holdout_RGB.pkl", "rb")))

# display image: used for testing
#cv.imshow('Window', cropped_image)
#cv.waitKey(0)
#cv.destroyAllWindows







# Reading in Training labels     
#y = np.genfromtxt("training_norm.csv", delimiter=",")[1:,1:]

















