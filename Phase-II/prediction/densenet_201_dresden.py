#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 23:58:15 2017

@author: root
"""
import tensorflow as tf
from keras.optimizers import SGD
from keras.layers import Input, merge, ZeroPadding2D, concatenate, Reshape, multiply
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K

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
import random
from dense_all_keras import DenseNetImageNet201

img_rows = 256
img_cols = 256
num_classes = 27

combined_train = list(zip(train_imdir, train_label))
random.shuffle(combined_train)

train_imdir[:], train_label[:] = zip(*combined_train)


combined_val= list(zip(val_imdir, val_label))
random.shuffle(combined_val)

val_imdir[:], val_label[:] = zip(*combined_val)

def augment(src, choice):
            
    if choice == 0:
        src = np.rot90(src, 1)
                
    if choice == 1:
        src = src
                
    if choice == 2:
        src = np.rot90(src, 2)
                
    if choice == 3:
        src = np.rot90(src, 3)
    return src

def get_patch(img,size):
    
    a = 256 - size
    d = np.random.randint(0,a)
    img2 = img[d:d+size,d:d+size,:]
    return img2
    
#def generate_processed_batch(inp_data,label,batch_size = 50, train = True):
#
#    batch_image1 = np.zeros((batch_size, img_rows, img_cols, 3))
#    batch_image2 = np.zeros((batch_size, 128, 128, 3))
#    batch_image3 = np.zeros((batch_size, 64, 64, 3))
#    batch_label = np.zeros((batch_size, 10))
#    if train==True:
#        num = 200000
#    else:
#        num = len(inp_data)
#    while 1:
#        for i_data in range(0,num,batch_size):
#            for i_batch in range(batch_size):
#                
#                img = imread(inp_data[i_data+i_batch])
#                img = augment(img, np.random.randint(4))
#                img1 = get_patch(img, 128)
#                img2 = get_patch(img, 64)
#
##                a = 256 - 128
##                d = np.random.randint(0,a)
##                img = img[d:d+img_rows,d:d+img_rows,:]
#            
#                
#                lab = np_utils.to_categorical(label[i_data+i_batch],10)
#
#                batch_image1[i_batch] = img
#                batch_image2[i_batch] = img1
#                batch_image3[i_batch] = img2
#                batch_label[i_batch] = lab
#
#            yield [batch_image1, batch_image2, batch_image3], batch_label
def generate_processed_batch(inp_data,label,batch_size = 50):

    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_label = np.zeros((batch_size, num_classes))
    while 1:
        for i_data in range(0,len(inp_data),batch_size):
            for i_batch in range(batch_size):
                if i_data + i_batch >= len(inp_data):
                    continue
                try:
                    img = imread(inp_data[i_data+i_batch])
                except Exception:
                    img = imread(inp_data[i_data+i_batch+1])
#                a = 256 - img_rows
#                d = np.random.randint(0,a)
#                img = img[d:d+img_rows,d:d+img_rows,:]
                try:
                    img = augment(img, np.random.randint(4))
                except Exception:
                    pass
                
                lab = np_utils.to_categorical(label[i_data+i_batch],num_classes)

                try:
                    batch_images[i_batch] = img
                except Exception:
                    print(i_data+i_batch)
                batch_label[i_batch] = lab

            yield batch_images, batch_label


def scheduler(epoch):
    
    if epoch!=0 and epoch%2 == 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr*.5)
        print("lr changed to {}".format(lr*.5))
            
    return K.get_value(model.optimizer.lr)

lr_decay = LearningRateScheduler(scheduler)

callbacks_list= [
    keras.callbacks.ModelCheckpoint(
        filepath='model_dresden_dkh_27_models_second_run.h5',
        mode='min',
        monitor='val_loss',
        save_best_only=True,
        verbose = 1
    ), lr_decay
]

batchsize = 16
#
training_gen = generate_processed_batch(train_imdir, train_label, batchsize)
val_gen = generate_processed_batch(val_imdir,val_label, batchsize)

def get_model():
    base_model = DenseNetImageNet201(input_shape = None,
                        bottleneck=True,
                        reduction=0.5,
                        dropout_rate=0.0,
                        weight_decay=1e-4,
                        include_top=False,
                        weights=None,
                        input_tensor=None,
                        pooling=None,
                        classes=None,
                        )
    x = base_model.output
    x= GlobalAveragePooling2D()(x)

    zz = Dense(10, activation = 'softmax')(x)
    model = Model(inputs = base_model.input, outputs= zz)
    model.load_weights('sp_all_data_dkh_2_run_at_lr_10_5.h5')
    
    
    model_dresden = Model(inputs = base_model.input, outputs = x)
    p = model_dresden.output
#    p = GlobalAveragePooling2D()(p)
    q = Dense(num_classes, activation = 'softmax')(p)
    
    model_final = Model(inputs = model_dresden.input, outputs= q)
#    model_new.trainable = False
    
    return model_final 


img_rows, img_cols = img_rows, img_rows # Resolution of inputs
channel = 3
num_classes = num_classes
#batch_size = batchsize
nb_epoch = 30

#batch_size = 16

#def squeeze_excite_block(input):
#    init = input
##    filters = init._keras_shape[channel_axis]
#    se_shape = ( 1, 1920)
#
#    se = GlobalAveragePooling1D()(init)
#    se = Reshape(se_shape)(se)
#    se = Dense(120, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
#    se = Dense(1920, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
#
#    x = multiply([init, se])
#    return x


# Load our model
#base_model = DenseNet201(include_top=False, weights='imagenet', input_shape= (img_rows, img_cols, channel), pooling=None, classes=None)

#x = base_model.output
#x= GlobalAveragePooling2D()(x)
#
#zz = Dense(10, activation = 'softmax')(x)
#
#model = Model(inputs = base_model.input, outputs= zz)
batch_size=16
model  = get_model()
model.summary()
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('model_dresden_dkh_27_models.h5')
model.fit_generator(training_gen,steps_per_epoch= len(train_imdir)/batch_size,nb_epoch=nb_epoch,validation_data=val_gen,
                    validation_steps=len(val_imdir)/batch_size,callbacks=callbacks_list,initial_epoch = 1)
