# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:38:07 2021

@author: Sarwan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:50:56 2021

@author: Sarwan
"""



# fetch the path to the test data

import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
import struct
import numpy as np
from skimage.transform import resize
from os import listdir
from os.path import isfile, join
import csv

#get filenames of all dicom images in the dataset folder
dataset_path = './dataset'
onlyfiles = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]

    
#we select minimum height and width for the images in the whole dataset
#this is done to make all images of same size
#so that we can make equal length feature vectors
min_width = 823	
min_height = 927

temp_matrix = np.zeros((len(onlyfiles),min_width*min_height))

for i in range(len(onlyfiles)):
#    print("i = ",i)
    image_path = './' + onlyfiles[i]
    
    dataset= pydicom.read_file(image_path)
    pixel_values = dataset.pixel_array
    
    #matrix of figure came here
    pixel_values = dataset.pixel_array
    type(pixel_values)
    pixel_values.shape
    
    
    
    resized_pixel_values = resize(pixel_values, (min_width, min_height), anti_aliasing=True)
    resized_pixel_values.shape
    
    resized_vector = resized_pixel_values.flatten()
    print("resized_vector dimensions = ",resized_vector.shape)
    print("Average = ",np.mean(resized_vector))
    temp_matrix[i,] = resized_vector
    print("Average mat row = ",np.mean(temp_matrix[i,]))


