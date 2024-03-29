#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 02:54:56 2017

@author: RR
"""
import tensorflow as tf
from keras.optimizers import SGD
from keras.layers import Input, merge, ZeroPadding2D, concatenate
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import log_loss
import keras
from keras.callbacks import LearningRateScheduler
from skimage.restoration import denoise_wavelet
from skimage import img_as_float,img_as_uint
from keras.utils import np_utils
import numpy as np
from scipy.misc import imread
#from imageio import imread
import math
from scipy import signal
import random
from dense_all_keras import DenseNetImageNet201
from keras.applications.densenet import DenseNet201


#img_rows = 128
#img_cols = 128
#num_classes = 4
#channel = 3
#batch_size= 16

img_rows = 256
img_cols = 256
num_classes = 4
channel = 3
batch_size= 4  #change batch size according to the size of the image

base_model = DenseNetImageNet201(input_shape = None,
                        bottleneck=True,
                        reduction=0.5,
                        dropout_rate=0.0,
                        weight_decay=1e-4,
                        include_top=False,
                        weights=None,
                        input_tensor=None,
                        pooling=None,
                        classes=None
                        )
x = base_model.output
x= GlobalAveragePooling2D()(x)

zz = Dense(4, activation = 'softmax')(x)
model = Model(inputs = base_model.input, outputs= zz)


model_final = Model(inputs = base_model.input, outputs = x)

rr = model_final.output
    
oo = Dense(4, activation = 'softmax')(rr)
    
final_model = Model(inputs = model_final.input, outputs= oo)
final_model.load_weights('phase_3_128_weight.h5')
#final_model.load_weights('phase_3_64_weight.h5')
#
#model_xception = load_model('Xception_Xlr.hdf5')

'''1=gamma, 2=jpeg, 3=resize,  4=unalt'''
def patch_creator(img):
    
    p=0;
    img_in = imread(img)

    m, n = img_in.shape[0:2]
    a, b = m//img_rows, n//img_cols
    all_patch = np.zeros((a*b, img_rows, img_cols, 3))
    for k in range(a):
        for l in range (b):
            all_patch[p,:,:,:] = img_in[(k*img_rows):(k+1)*img_rows, (l*img_cols):(l+1)*img_cols, :]
            p+=1
            
    return all_patch
y_gen2 = np.zeros((len(test_imdir),1))
y_pred_2 = np.zeros((len(test_imdir),batch_size,4))

for i in range(len(test_imdir)):
    patch = patch_creator(test_imdir[i])
    y_pred_2[i] = final_model.predict(patch, batch_size = batch_size)
    print(i)
y_pred2 = np.average(y_pred_2,axis = 1)
y_gen2 = y_pred2.argmax(axis=1)+1
result = np.bincount(y_gen2) 

