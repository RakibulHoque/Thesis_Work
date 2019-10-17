#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:05:13 2017

@author: RR
"""
#test_dir='G:\\RR\\test_unalt_manip\\*\\'
test_dir = 'D:\\RR\\test_unalt_manip\\gamma*\\'

#manip =  ['unalt', 
#          'resize_0.5',
#          'resize_0.8',
#          'resize_1.5',
#          'resize_2.0',
#          'gamma_0.8',
#          'gamma_1.2',
#          'jpeg_70',
#          'jpeg_90']

#model = ['galaxynote', 'galaxys', 'htc', 'iphone4', 
#         'iphone6', 'lg', 'motodroid', 
#         'motonex', 'motox', 'sony']
          
import random
import cv2, glob, numpy as np, os
from sklearn.feature_extraction import image

test_imdir=[]
test_label=[]

images = glob.glob(test_dir +'/*')

test_imdir=test_imdir+images