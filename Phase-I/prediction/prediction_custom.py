#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 02:54:56 2017

@author: root
"""
import tensorflow as tf
from keras.optimizers import SGD
from keras.layers import Input, merge, ZeroPadding2D, concatenate
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as Kyyfffffffff
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from dense_all_keras import DenseNetImageNet201
from sklearn.metrics import log_loss
#from scale_layers import Scale
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
from statistics import mode
import random
#from dense_all_keras import DenseNetImageNet201
from keras.applications.densenet import DenseNet201

img_rows = 64
img_cols = 64
num_classes = 10
channel = 3
batch_size=64

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

zz = Dense(10, activation = 'softmax')(x)
#model = Model(inputs = base_model.input, outputs= zz)


model_final = Model(inputs = base_model.input, outputs = x)
#model_final.load_weights('densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')            
#model_final.trainable = False

rr = model_final.output
    
oo = Dense(10, activation = 'softmax')(rr)
    
final_model = Model(inputs = model_final.input, outputs= oo)

final_model.load_weights('64_paper.h5')
#
#model_xception = load_model('Xception_Xlr.hdf5')


def patch_creator(img):
    
    p=0;
    img_in = imread(img)
#    img = img_as_float(img_in)
#    img = img/255.0
#    img = denoise_wavelet(img, multichannel=True, mode='hard')
#    img = img_as_uint(img)
    m, n = img_in.shape[0:2]                                                                                                                                                  
    a, b = 512//img_rows, 512//img_cols
    all_patch = np.zeros((a*b, img_rows, img_cols, 3))
    for k in range(a):
        for l in range (b):
            all_patch[p,:,:,:] = img_in[(k*img_rows):(k+1)*img_rows, (l*img_cols):(l+1)*img_cols, :]
            p+=1
            
    return all_patch

y_gen2 = np.zeros((len(test_imdir),1))
y_pred_2 = np.zeros((len(test_imdir),batch_size,10))
for i in range(len(test_imdir)):
    
    patch = patch_creator(test_imdir[i])  
#    y_pred_1 = model_resnet.predict(patch, batch_size = 25)
    y_pred_2[i] = final_model.predict(patch, batch_size = batch_size)
#    y_pred_3 = model_xception.predict(patch, batch_size = 25)
    
#    y_final = (y_pred_1 + y_pred_2 + y_pred_3)/3
    
    
#    y_pred_2[i] = y_pred_2[np.max(y_pred_2,axis=1)>0.5]
#    y_pred2=y_pred_2[i].argmax(axis=1)+1;
#    y_pred2 = np.average(y_pred_2,axis=0)
#    y_gen2[i,0] = mode(y_pred2)
    
    
#    y_gen2_xception[i] = y_pred2.argmax(axis=1)+1
    print(i)
y_pred2 = np.average(y_pred_2,axis = 1)
y_gen2 = y_pred2.argmax(axis=1)+1
 
#fun = np.copy(y_gen2)
#fun.dump('improve.dat')
#
#for i in range(len(test_imdir)):
#    if y_gen2[i] != 2 :
#        os.remove(test_imdir[i])