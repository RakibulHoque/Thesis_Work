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
#from skimage.restoration import denoise_wavelet
#from skimage import img_as_float,img_as_uint
from keras.utils import np_utils
import numpy as np
#from scipy.misc import imread
from imageio import imread
import math
from scipy import signal
import random
from dense_all_keras import DenseNetImageNet201

total_image = len(train_imdir)
img_rows = 64
img_cols = 64

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

def generate_processed_batch(inp_data,label,batch_size = 50):

    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_label = np.zeros((batch_size, 4))
    while 1:
#        d = np.random.randint(0,1500000)
        combined_train = list(zip(inp_data, label))
        random.shuffle(combined_train)
        inp_data[:], label[:] = zip(*combined_train)        
        for i_data in range(0,total_image,batch_size):
            for i_batch in range(batch_size):

                img = imread(inp_data[i_data+i_batch])
                img = augment(img, np.random.randint(4))
                
                a = 256 - img_rows
                d = np.random.randint(0,a)
                img = img[d:d+img_rows,d:d+img_rows,:]
                
                lab = np_utils.to_categorical(label[i_data+i_batch],4)

                batch_images[i_batch] = img
                batch_label[i_batch] = lab

            yield batch_images, batch_label


def generate_processed_batch_val(inp_data,label,batch_size = 50):

    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_label = np.zeros((batch_size, 4))
    while 1:
        for i_data in range(0,len(inp_data),batch_size):
            for i_batch in range(batch_size):
#                print(i_data+i_batch)
                img = imread(inp_data[i_data+i_batch])
                img = augment(img, np.random.randint(4))
                
                a = 256 - img_rows
                d = np.random.randint(0,a)
                img = img[d:d+img_rows,d:d+img_rows,:]
                
                lab = np_utils.to_categorical(label[i_data+i_batch],4)

                batch_images[i_batch] = img
                batch_label[i_batch] = lab

            yield batch_images, batch_label




def scheduler(epoch):
    
    if epoch!=0 and epoch%2 == 0:
        lr = K.get_value(final_model.optimizer.lr)
        K.set_value(final_model.optimizer.lr, lr*.5)
        print("lr changed to {}".format(lr*.5))
            
    return K.get_value(final_model.optimizer.lr)

lr_decay = LearningRateScheduler(scheduler)

callbacks_list= [
    keras.callbacks.ModelCheckpoint(
        filepath='LAST_RUN_{epoch}.h5',
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

batchsize = 64

training_gen = generate_processed_batch(train_imdir, train_label, batchsize)
val_gen = generate_processed_batch_val(val_imdir,val_label, batchsize)
    



img_rows, img_cols = img_rows, img_rows # Resolution of inputs
channel = 3
num_classes = 4
batch_size = batchsize
nb_epoch = 20

n = batchsize
from densenet_201_new import get_model

#def notun_model():
#    
#    model = get_model()
##    model.trainable = False
#    x = model.output
#    x= GlobalAveragePooling2D()(x)
#    zz = Dense(10, activation = 'softmax')(x)
#    model = Model(inputs = model.input  , outputs = zz)
#    
#    return model

#model = notun_model()

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
model = Model(inputs = base_model.input, outputs= zz)

model.load_weights('sp_all_data_dkh_2_run_at_lr_10_5_second_time_run_on_64.h5')
#model.trainable = False
model_final = model.get_layer(index = -2).output#Model(inputs = model.input, outputs = x)
#model_final.load_weights('densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')       
#model_final.trainable = False

#rr = model_final.output
    
#model_final.trainable = False 

oo = Dense(4, activation = 'softmax')(model_final)
    
final_model = Model(inputs = model.input, outputs= oo)

i=0
for dkh in final_model.layers:
    i+=1
    if i==len(final_model.layers):
        break
    dkh.trainable = False
    
    
final_model.summary()

sgd = SGD(lr=1e-03, decay=1e-6, momentum=0.9, nesterov=True)
final_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
#final_model.load_weights('desne_process_13.h5')
final_model.fit_generator(training_gen,steps_per_epoch=int(total_image/n),nb_epoch=nb_epoch,validation_data=val_gen,
                    validation_steps=int(len(val_imdir)/n),callbacks=callbacks_list,
                    initial_epoch=0 )