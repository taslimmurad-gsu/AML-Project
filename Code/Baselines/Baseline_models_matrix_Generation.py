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


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

#Import scikit-learn dataset library
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
#Load dataset
# cancer = datasets.load_breast_cancer()
from sklearn import svm
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt

# In[0]
#get filenames of all dicom images in the dataset folder
#dataset_path = 'D:/University/Ph.D/Advanced Machine Learning/Project/Code/dataset'
dataset_path = 'G:/Train Data/'
onlyfiles = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
rows_number = 200
onlyfiles = onlyfiles[1:rows_number]


train_data = pd.read_csv (r'D:/University/Ph.D/Advanced Machine Learning/Project/Code/train.csv')
# 0 index has image ID
# 2 index has class id


# In[0]
#we select minimum height and width for the images in the whole dataset
#this is done to make all images of same size
#so that we can make equal length feature vectors
min_width = 823	
min_height = 927

temp_length = min_width*min_height+1
temp_matrix = np.zeros((len(onlyfiles),min_width*min_height))

feature_matrix = []
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
    feature_matrix.append(resized_vector)
    print("resized_vector dimensions = ",resized_vector.shape)
    print("Average = ",np.mean(resized_vector))
    temp_matrix[i,] = resized_vector
    print("Average mat row = ",np.mean(temp_matrix[i,]))
    
# print("feature_matrix shape = ", feature_matrix)
# Write to the pdf
#df = pd.DataFrame(temp_matrix)
#
#df.to_csv (r'D:/University/Ph.D/Advanced Machine Learning/Project/Code/Preprocessed_Original.csv', index = False, header=True)
# In[1]

train_data2 = train_data.to_numpy()
dataset_path = 'G:/Train Data/'
onlyfiles = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
onlyfiles_names = onlyfiles[1:rows_number]

# In[2]
onlyfiles_names2 = onlyfiles_names
for i in range(len(onlyfiles_names)):
    print("i = ",i)
    onlyfiles_names2[i] = onlyfiles_names[i].replace('.dicom','')

true_label = onlyfiles_names

# In[3]
for i in range(len(onlyfiles_names)):
    print("i = ",i)
    check_temp = np.where(onlyfiles_names2[i] == train_data2[:,0])
    check_temp2 = np.asarray(check_temp)
    true_label[i] = train_data2[check_temp2[0,0],2]

true_label3 = true_label

true_label = true_label3
# In[3]
##################### PCA Dimensionality Reduction Logic Starts Here ########

pca2 = PCA(n_components=30)
pca2.fit(feature_matrix)
x_3d = pca2.transform(feature_matrix)


#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(x_3d, true_label, 
                                                    test_size=0.3, random_state=0)

pca_variance = pca2.explained_variance_

plt.figure(figsize=(8, 6))
plt.plot(pca_variance,label='individual variance')
plt.legend()
plt.ylabel('Variance ratio')
plt.xlabel('Principal components')
plt.show()

##################### PCA Dimensionality Reduction Logic ends Here ########


# In[3]
##################### Lasso Dimensionality Reduction Logic Starts Here ########

X_train, X_test, y_train, y_test = train_test_split(
    pd.DataFrame(feature_matrix),true_label,
    test_size=0.3,
    random_state=0)
X_train.shape, X_test.shape

scaler = StandardScaler()
scaler.fit(X_train.fillna(0))

sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l2', solver='liblinear'))
sel_.fit(scaler.transform(X_train.fillna(0)), y_train)

sel_.get_support()

selected_feat = X_train.columns[(sel_.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
      np.sum(sel_.estimator_.coef_ == 0)))

X_train_selected = sel_.transform(X_train.fillna(0))
X_test_selected = sel_.transform(X_test.fillna(0))
X_train_selected.shape, X_test_selected.shape

X_train = X_train_selected
X_test = X_test_selected
# y_train
# y_test

##################### Lasso Dimensionality Reduction Logic ends Here ########

# In[0]
##################### Correlation Plot for features starts Here ########
# tmp_val = 3000
final_data = np.concatenate((X_train, X_test), axis=0)
asd = pd.DataFrame(final_data)
data_path = "D:/University/Ph.D/Advanced Machine Learning/Project/Code/rodge_correlation_matrix.npy"
np.save(data_path, asd)
# sns.heatmap(asd.corr());

# dataframe.corr()
##################### Correlation Plot for features starts Here ########


# In[0]
##########################  SVM Classifier  ################################
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)



# Model Accuracy: how often is the classifier correct?
svm_acc = metrics.accuracy_score(y_test, y_pred)
print("SVM Accuracy:",svm_acc)
svm_prec = metrics.precision_score(y_test, y_pred,average='weighted')
# Model Precision: what percentage of positive tuples are labeled as such?
print("SVM Precision:",svm_prec)
svm_recall = metrics.recall_score(y_test, y_pred,average='weighted')
# Model Recall: what percentage of positive tuples are labelled as such?
print("SVM Recall:",svm_recall)
svm_f1 = metrics.f1_score(y_test, y_pred,average='weighted')
# Model Recall: what percentage of positive tuples are labelled as such?
print("SVM F1:",svm_f1)


# In[0]
##########################  NB Classifier  ################################
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
# print("Number of mislabeled points out of a total %d points : %d" 
#       % (X_test.shape[0], (y_test != y_pred).sum()))

# Model Accuracy: how often is the classifier correct?
NB_acc = metrics.accuracy_score(y_test, y_pred)
print("Gaussian NB Accuracy:",NB_acc)
NB_prec = metrics.precision_score(y_test, y_pred,average='weighted')
# Model Precision: what percentage of positive tuples are labeled as such?
print("Gaussian NB Precision:",NB_prec)
NB_recall = metrics.recall_score(y_test, y_pred,average='weighted')
# Model Recall: what percentage of positive tuples are labelled as such?
print("Gaussian NB Recall:",NB_recall)
NB_f1 = metrics.f1_score(y_test, y_pred,average='weighted')
# Model Recall: what percentage of positive tuples are labelled as such?
print("Gaussian NB F1:",NB_f1)


# In[0]
##########################  MLP Classifier  ################################
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 
# Feature scaling
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test_2 = scaler.transform(X_test)


# Finally for the MLP- Multilayer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
mlp.fit(X_train, y_train)


y_pred = mlp.predict(X_test_2)

# print(predictions)
# # Last thing: evaluation of algorithm performance in classifying flowers
# print(confusion_matrix(y_test,predictions))  
# print(classification_report(y_test,predictions))

# Model Accuracy: how often is the classifier correct?
MLP_acc = metrics.accuracy_score(y_test, y_pred)
print("MLP Accuracy:",MLP_acc)
MLP_prec = metrics.precision_score(y_test, y_pred,average='weighted')
# Model Precision: what percentage of positive tuples are labeled as such?
print("MLP Precision:",MLP_prec)
MLP_recall = metrics.recall_score(y_test, y_pred,average='weighted')
# Model Recall: what percentage of positive tuples are labelled as such?
print("MLP Recall:",MLP_recall)
MLP_f1 = metrics.f1_score(y_test, y_pred,average='weighted')
# Model Recall: what percentage of positive tuples are labelled as such?
print("MLP F1:",MLP_f1)


# In[0]
##########################  knn Classifier  ################################

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Model Accuracy: how often is the classifier correct?
knn_acc = metrics.accuracy_score(y_test, y_pred)
print("Knn Accuracy:",knn_acc)
knn_prec = metrics.precision_score(y_test, y_pred,average='weighted')
# Model Precision: what percentage of positive tuples are labeled as such?
print("Knn Precision:",knn_prec)
knn_recall = metrics.recall_score(y_test, y_pred,average='weighted')
# Model Recall: what percentage of positive tuples are labelled as such?
print("Knn Recall:",knn_recall)
knn_f1 = metrics.f1_score(y_test, y_pred,average='weighted')
# Model Recall: what percentage of positive tuples are labelled as such?
print("Knn F1:",knn_f1)


# In[0]
##########################  Logistic Regression Classifier  ################################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Accuracy: how often is the classifier correct?
LR_acc = metrics.accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:",LR_acc)
LR_prec = metrics.precision_score(y_test, y_pred,average='weighted')
# Model Precision: what percentage of positive tuples are labeled as such?
print("Logistic Regression Precision:",LR_prec)
LR_recall = metrics.recall_score(y_test, y_pred,average='weighted')
# Model Recall: what percentage of positive tuples are labelled as such?
print("Logistic Regression Recall:",LR_recall)
LR_f1 = metrics.f1_score(y_test, y_pred,average='weighted')
# Model Recall: what percentage of positive tuples are labelled as such?
print("Logistic Regression F1:",LR_f1)


# In[0]
##########################  Random Forest Classifier  ################################
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 100)
# Train the model on training data
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
fr_acc = metrics.accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:",fr_acc)
fr_prec = metrics.precision_score(y_test, y_pred,average='weighted')
# Model Precision: what percentage of positive tuples are labeled as such?
print("Random Forest Precision:",fr_prec)
fr_recall = metrics.recall_score(y_test, y_pred,average='weighted')
# Model Recall: what percentage of positive tuples are labelled as such?
print("Random Forest Recall:",fr_recall)
fr_f1 = metrics.f1_score(y_test, y_pred,average='weighted')
# Model Recall: what percentage of positive tuples are labelled as such?
print("Random Forest F1:",fr_f1)
