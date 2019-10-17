#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 23:58:15 2017

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
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import log_loss
#from scale_layers import Scale
import keras
from keras.callbacks import LearningRateScheduler
from skimage.restoration import denoise_wavelet
from skimage import img_as_float,img_as_uint
from keras.utils import np_utils
import numpy as np
#from scipy.misc import imread
from imageio import imread
import math
from scipy import signal
import random
from dense_all_keras import DenseNetImageNet201
from densenet_201_new import get_model

#
#img_cols1 = 256

img_rows = 256
img_cols = 256

#img_rows2 = 128
#img_cols2 = 128

def get_all_patch(img, size):
    num = int(256//size)
    all_patch = np.zeros((num*num, size, size, 3))
    p=0
    for i in range(num):
        for j in range(num):
            all_patch[p]=img[i+size,j+size,:]
            p+=1
    return all_patch

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

from tqdm import tqdm

def generate_processed_batch(inp_data,label,batch_size = 50):

    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_label = np.zeros((batch_size, 10))
    while 1:
        for i_data in range(0,len(inp_data),batch_size):
            for i_batch in range(batch_size):
                print(i_data+i_batch)
                img = imread(inp_data[i_data+i_batch])
                img = augment(img, np.random.randint(4))
                
                a = 256 - img_rows
                d = np.random.randint(0,a)
                img = img[d:d+img_rows,d:d+img_rows,:]
                
                lab = np_utils.to_categorical(label[i_data+i_batch],10)

                batch_images[i_batch] = img
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
        filepath='l2.h5',
        mode='min',
        monitor='val_loss',
        save_best_only=True,
        verbose = 1
    ), lr_decay]
            
#            
#    EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-3),
#    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=1, 
#                               verbose=1, min_lr=1e-7)
#]

batchsize = 16

#training_gen = generate_processed_batch(train_imdir, train_label, batchsize)
#val_gen = generate_processed_batch(val_imdir,val_label, batchsize)
    



img_rows, img_cols = img_rows, img_rows # Resolution of inputs
channel = 3
num_classes = 10
batch_size = batchsize
nb_epoch = 30

n = batchsize
from keras.applications.densenet import DenseNet201
def purana_model(wt_path):
    
    base_model = DenseNet201(include_top=False, weights='imagenet', input_shape= (img_rows, img_cols, channel), 
                             pooling=None, classes=None) 
    x = base_model.output
    y = GlobalAveragePooling2D()(x)
    zz = Dense(10, activation = 'softmax')(y)
    
    model = Model(inputs = base_model.input, outputs= zz)
    model.load_weights(wt_path)
    
    
    model_new = Model(inputs = base_model.input, outputs = y)

    
    return model_new
    

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
#
#model.load_weights('sp_all_data_dkh_2_run_at_lr_10_5_second_time.h5')
#model.load_weights('just_emd_201_starter.h5')
#    mod = Model(inputs = model.input, outputs = model.get_layer(index=-2).output)
model_64_path = 'sp_all_data_dkh_2_run_at_lr_10_5_second_time_run_on_64.h5'
model_64 = get_model(model_64_path)
model_128_path = 'sp_all_data_dkh_2_run_at_lr_10_5_second_time_run_on_128.h5'
model_128 = get_model(model_128_path)
model_256_path = 'sp_all_data_dkh_2.h5'
model_256 = purana_model(model_256_path)

model_256_feat = np.zeros((200000,1920))
model_128_feat = np.zeros((200000,1920))
model_64_feat = np.zeros((200000,1920))

print('start')

for i in tqdm(range(200000)):
    img = imread(train_imdir[i])
    img_128 = get_all_patch(img,128)
    img_64 = get_all_patch(img, 64)
    img_256 = np.expand_dims(img, axis= 0)
    model_256_feat[i] = model_256.predict(img_256)
    model_128_feat[i] = np.average(model_128.predict_on_batch(img_128), axis = 0)
    model_64_feat[i] = np.average(model_64.predict_on_batch(img_64), axis = 0)
#model2 = get_model()
#
#model3 = purana_model()
#model1_out = np.zeros((len(train_imdir), 1920))

#for i in range(len(train_imdir)):
#    if(i%1000):
#        print(i)

#print('dhukse\n')
#
#model1_out = model1.predict_generator(training_gen, steps = 200000/16)
#
#print('ber hoise\n')

np.save('model_128_feat', model_128_feat)
np.save('model_64_feat', model_64_feat)
np.save('model_256_feat', model_256_feat)
np.save('l2_model_label', train_label)
    

#m2 = model2.get_layer(index = -2).output
#m2 = Reshape(( 1, 1920))(m2)
#
#m3 = model3.get_layer(index = -2).output
#m3 = Reshape(( 1, 1920))(m3)
#
#mall = concatenate([m1,m2,m3], axis = -2)
#x= GlobalAveragePooling1D()(mall)
#zz = Dense(10, activation = 'softmax')(x)
#model1.summary()
#sgd = SGD(lr=1e-03, decay=1e-6, momentum=0.9, nesterov=True)
#model1.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
#
#model2.summary()
#sgd = SGD(lr=1e-03, decay=1e-6, momentum=0.9, nesterov=True)
#model2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
#
#model3.summary()
#sgd = SGD(lr=1e-03, decay=1e-6, momentum=0.9, nesterov=True)
#model3.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

#model_all = concatenate([])

#model.load_weights('sp_all_data_dkh_2_run_at_lr_10_5_second_time.h5')
#model.fit_generator(training_gen,steps_per_epoch=int(len(train_imdir)/n),nb_epoch=nb_epoch,validation_data=val_gen,
#                    validation_steps=int(len(val_imdir)/n),callbacks=callbacks_list)
