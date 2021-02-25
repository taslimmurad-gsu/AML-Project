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
import numpy as np
import struct
from os import listdir
from os.path import isfile, join
import csv
import pandas as pd

from skimage.transform import resize


from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression



#get filenames of all dicom images in the dataset folder
#dataset_path = 'D:/University/Ph.D/Advanced Machine Learning/Project/Code/dataset'
dataset_path = 'G:/Train Data/'
onlyfiles = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
onlyfiles = onlyfiles[1:1000]


train_data = pd.read_csv (r'D:/University/Ph.D/Advanced Machine Learning/Project/Code/train.csv')
# 0 index has image ID
# 2 index has class id


#we select minimum height and width for the images in the whole dataset
#this is done to make all images of same size
#so that we can make equal length feature vectors
min_width = 823	
min_height = 927

temp_length = min_width*min_height+1
temp_matrix = np.zeros((len(onlyfiles),min_width*min_height))

for i in range(len(onlyfiles)):
    print("i = ",i)
    image_path = dataset_path + onlyfiles[i]
    
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
    

# Write to the pdf
#df = pd.DataFrame(temp_matrix)
#
#df.to_csv (r'D:/University/Ph.D/Advanced Machine Learning/Project/Code/Preprocessed_Original.csv', index = False, header=True)
