"""
Created on Sat Dec 30 15:41:11 2017

@author: DSP
"""
import math
import numpy as np
import os
import glob
import time
from imageio import imwrite
from PIL import Image
from operator import itemgetter
import matplotlib.pyplot as plt
start = time.time()
#image_dir = "D:\\RR\\Pyworks\\Co_occurance\\1.tif"


def patch_creator(single_image_dir,num_row = 128,num_col = 128):
    im = Image.open(single_image_dir)
    single_image = np.array(im)
    a , b = single_image.shape[0], single_image.shape[1]
    k , m = a//num_row, b//num_col
    ind_row = np.repeat(range(0,k*num_row,num_row), m)
    ind_col = np.tile(range(0,m*num_col,num_col), k)
    image_patches = [single_image[a1:a1+num_row,a2:a2+num_col,:] for (a1,a2) in zip(ind_row,ind_col)]    
    return image_patches

def find_quality(patches, sel_no = 3):
    alpha = 0.7
    beta = 4
    gamma = math.log(0.01)    
    Constant_1 = np.repeat(np.array([alpha]),3)*np.repeat(np.array([beta]),3)
    Constant_2 = np.repeat(np.array([1]),3)-np.repeat(np.array([alpha]),3)
    Constant_3 = np.repeat(np.array([1]),3)  
    zipped = []
    quality = []
    for i in patches:
        img = i/255.0
        chnl_mean = np.mean(img, axis=(0,1))    
        chnl_std = np.std(img, axis=(0,1), ddof = 1)
        part_1 = Constant_1*(chnl_mean - chnl_mean*chnl_mean)
        part_2 = Constant_2*(Constant_3 - np.exp(np.repeat(np.array([gamma]),3)*chnl_std))
        img_qulty = np.mean(part_1 + part_2)    
        quality.append(img_qulty)
    zipped = zip(quality, patches)
    zipped = sorted(zipped,key=itemgetter(0))
    best_patches = zipped[-sel_no:]
    return best_patches


def saveimg(wrt_dir, img_counter, zipped_patches):
    k = 0  
    for i in zipped_patches:  
        image_wrt_dir = wrt_dir  +'//{}_{}.tif'.format(img_counter, k)
        imwrite(image_wrt_dir,i[1])
        k += 1    

#model = ['HTC-1-M7', 'iPhone-4s', 'iPhone-6', 'LG-Nexus-5x', 
#         'Motorola-Droid-Maxx', 'Motorola-Nexus-6', 'Motorola-X', 
#         'Samsung-Galaxy-Note3', 'Samsung-Galaxy-S4', 'Sony-NEX-7']
folders = ['resize_0.5','resize_0.8','resize_1.5','resize_2.0','gamma_0.8','gamma_1.2','jpeg_70','jpeg_90','unalt']
model = ['htc', 'iphone4', 'iphone6', 'lg', 
         'motodroid', 'motonex', 'motox', 
         'samsung_galaxynote', 'samsung_galaxys', 'sony']
imreadpath = 'F:\\RR\\Collected_Validation\\' 
#imreadpath = 'F:\\RR\\Collected_Validation\\' 
imwritepath = 'F:\\RR\\kaggle_validation_256\\' 

num_row = 256
num_col = 256
sel_no = 20
folder = 8
for m in range(len(model)):
    images = glob.glob(imreadpath + folders[folder] + '\\'+   model[m] + '\\' +'/*')
    os.makedirs(imwritepath+folders[folder]+'\\'+model[m]+'\\')
    img_no = 0
    for img_name in images:
        img_no += 1
        patches = patch_creator(img_name,num_row = num_row,num_col = num_col)
        selected_patches = find_quality(patches,sel_no = sel_no)
#        image_wrt_dir = imwritepath+ model[m]  +'/{}.png'.format(img_name.split(os.sep)[-1].split('.')[0])
        k = 0  
        for img in selected_patches:  
            k += 1
            image_wrt_dir = imwritepath +folders[folder]+'\\'+model[m]  +'/{}_{}_{}.png'.format(model[m],img_name.split(os.sep)[-1].split('.')[0],k)
#            img[1].dump(image_wrt_dir)
            imwrite(image_wrt_dir,img[1])
        print(model[m]+"_"+str(img_no))
 
#        saveimg(imwritepath + model[m], img_no, selected_patches) 
#        imwrite(image_wrt_dir,selected_patches[0][1])
    
#A = patch_creator(image_dir,num_row = 64,num_col = 64)
#W = find_quality(A,sel_no=30)
#image_wrt_dir = 'D://RR//Pyworks//Co_occurance//foo'
#
#saveimg(image_wrt_dir,W)
#print(time.time()-start)