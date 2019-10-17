#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 04:17:01 2017
@author: root
"""

#train_dir='E:\\RR\\centre_blocked_32_size\\'

train_dir = 'G:\\RR\\LAST_HOPE_DATASET\\'

model = ['*gamma*', '*jpeg*', '*resize*', '*unalt']

import random
import glob
import numpy as np
#import cv2, glob, numpy as np, os
from sklearn.feature_extraction import image
#

val_imdir=[]
val_label=[]

#val_saturated_imdir=[]
#val_saturated_label=[]
#val_smooth_imdir=[]
#val_smooth_label=[]
#

train_imdir=[]
train_label=[]

#train_saturated_imdir=[]
#train_saturated_label=[]
#train_smooth_imdir=[]
#train_smooth_label=[]


for i in range(len(model)):
#    images_others = glob.glob(train_dir + model[i] + '/*_others.dat')
#    images_smooth = glob.glob(train_dir + model[i] + '/*_saturated.dat')
#    images_saturated = glob.glob(train_dir + model[i] + '/*_smooth.dat')
#    images_all = images_smooth + images_saturated
    images_all = glob.glob(train_dir + model[i] + '\\*\\*')
#    random.shuffle(images_others)
#    random.shuffle(images_smooth)
    random.shuffle(images_all)
    
    a = len(images_all) * 0.99
    a = np.floor(len(images_all) * 0.99)
    a = int(a)
    b = a %64
    a = a - b
    c = len(images_all)
    d = c %64
    c = c - d
    
    images_all_train = images_all[0:100000]
    images_all_val = images_all[100000:104992]
    
#    images_smooth_train = images_smooth[0:2000]
#    images_smooth_val = images_smooth[2000:2250]
#    images_saturated_train = images_saturated[0:5000]
#    images_saturated_val = images_saturated[5000:5500]
    label_train=[i]*len(images_all_train)
    label_val = [i]*len(images_all_val)
#    label_train_smooth=[i]*len(images_smooth_train)
#    label_val_smooth = [i]*len(images_smooth_val)
#    label_train_saturated=[i]*len(images_saturated_train)
#    label_val_saturated = [i]*len(images_saturated_val)
#    
    train_imdir=train_imdir+images_all_train
    train_label=train_label+label_train
#    train_saturated_imdir=train_saturated_imdir+images_saturated_train
#    train_saturated_label=train_saturated_label+label_train_saturated
#    train_smooth_imdir=train_smooth_imdir+images_smooth_train
#    train_smooth_label=train_smooth_label+label_train_smooth

    val_imdir=val_imdir+images_all_val
    val_label=val_label+label_val
#    val_saturated_imdir=val_saturated_imdir+images_saturated_val
#    val_saturated_label=val_saturated_label+label_val_saturated
#    val_smooth_imdir=val_smooth_imdir+images_smooth_val
#    val_smooth_label=val_smooth_label+label_val_smooth
##    
    
    
    
    