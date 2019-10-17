#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:05:13 2017

@author: root
"""
test_dir='E:\\RR\\test\\'

#model = ['galaxynote', 'galaxys', 'htc', 'iphone4', 
#         'iphone6', 'lg', 'motodroid', 
#         'motonex', 'motox', 'sony']

import random
import glob, numpy as np, os
from sklearn.feature_extraction import image
#
#val_imdir=[]
#val_label=[]
#
test_imdir=[]
test_label=[]

images = glob.glob(test_dir +'/*')
#    random.shuffle(images)
#    images = images[0:10000]
    #images = images[0:500]
#    label=[i]*len(images)
#
test_imdir=test_imdir+images
#    test_label=test_label+label

#    val_imdir=val_imdir+images
#    val_label=val_label+label
##
    
    