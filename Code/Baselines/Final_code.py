# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 20:39:36 2021

@author: Sarwan
"""

#import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
import numpy as np
import struct
from os import listdir
from os.path import isfile, join
import csv
import pandas as pd


from skimage import data, color
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
from sklearn import svm
from sklearn import metrics

#import seaborn as sns

from pandas import DataFrame

from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score



from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# In[0]

#get filenames of all dicom images in the dataset folder
#dataset_path = 'D:/University/Ph.D/Advanced Machine Learning/Project/Code/dataset'
dataset_path = 'E:/Train Data/'
onlyfiles = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
rows_number = 15000
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

final_orig_data = temp_matrix
# In[1]
train_data2 = train_data.to_numpy()
dataset_path = 'E:/Train Data/'
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


# In[4]
def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc
    
    
  check = pd.DataFrame(roc_auc_dict.items())
  return mean(check)

# In[5]
##########################  SVM Classifier  ################################
def svm_fun(X_train,y_train,X_test,y_test):
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    svm_acc = metrics.accuracy_score(y_test, y_pred)
#     print("SVM Accuracy:",svm_acc)
    
    svm_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("SVM Precision:",svm_prec)
    
    svm_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("SVM Recall:",svm_recall)

    svm_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("SVM F1 Weighted:",svm_f1_weighted)
    
    svm_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("SVM F1 macro:",svm_f1_macro)
    
    svm_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("SVM F1 micro:",svm_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix SVM : \n", confuse)
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
#    print(macro_roc_auc_ovo[1])
    check = [svm_acc,svm_prec,svm_recall,svm_f1_weighted,svm_f1_macro,svm_f1_micro,macro_roc_auc_ovo[1]]
    return(check)
    


# In[5]
##########################  NB Classifier  ################################
def gaus_nb_fun(X_train,y_train,X_test,y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)


    NB_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Gaussian NB Accuracy:",NB_acc)

    NB_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB Precision:",NB_prec)
    
    NB_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB Recall:",NB_recall)
    
    NB_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB F1 weighted:",NB_f1_weighted)
    
    NB_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Gaussian NB F1 macro:",NB_f1_macro)
    
    NB_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Gaussian NB F1 micro:",NB_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix NB : \n", confuse)
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    check = [NB_acc,NB_prec,NB_recall,NB_f1_weighted,NB_f1_macro,NB_f1_micro,macro_roc_auc_ovo[1]]
    return(check)

# In[5]
##########################  MLP Classifier  ################################
def mlp_fun(X_train,y_train,X_test,y_test):
    # Feature scaling
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)  
    X_test_2 = scaler.transform(X_test)


    # Finally for the MLP- Multilayer Perceptron
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
    mlp.fit(X_train, y_train)


    y_pred = mlp.predict(X_test_2)
    
    MLP_acc = metrics.accuracy_score(y_test, y_pred)
#     print("MLP Accuracy:",MLP_acc)
    
    MLP_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("MLP Precision:",MLP_prec)
    
    MLP_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("MLP Recall:",MLP_recall)
    
    MLP_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("MLP F1:",MLP_f1_weighted)
    
    MLP_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("MLP F1:",MLP_f1_macro)
    
    MLP_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("MLP F1:",MLP_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix MLP : \n", confuse)
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [MLP_acc,MLP_prec,MLP_recall,MLP_f1_weighted,MLP_f1_macro,MLP_f1_micro,macro_roc_auc_ovo[1]]
    return(check)

# In[5]
##########################  knn Classifier  ################################
def knn_fun(X_train,y_train,X_test,y_test):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    knn_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Knn Accuracy:",knn_acc)
    
    knn_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Knn Precision:",knn_prec)
    
    knn_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Knn Recall:",knn_recall)
    
    knn_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Knn F1 weighted:",knn_f1_weighted)
    
    knn_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Knn F1 macro:",knn_f1_macro)
    
    knn_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Knn F1 micro:",knn_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix KNN : \n", confuse)
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [knn_acc,knn_prec,knn_recall,knn_f1_weighted,knn_f1_macro,knn_f1_micro,macro_roc_auc_ovo[1]]
    return(check)

# In[5]
##########################  Random Forest Classifier  ################################
def rf_fun(X_train,y_train,X_test,y_test):
    # Import the model we are using
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators = 100)
    # Train the model on training data
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    fr_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Random Forest Accuracy:",fr_acc)
    
    fr_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Random Forest Precision:",fr_prec)
    
    fr_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Random Forest Recall:",fr_recall)
    
    fr_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Random Forest F1 weighted:",fr_f1_weighted)
    
    fr_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Random Forest F1 macro:",fr_f1_macro)
    
    fr_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Random Forest F1 micro:",fr_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix RF : \n", confuse)
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [fr_acc,fr_prec,fr_recall,fr_f1_weighted,fr_f1_macro,fr_f1_micro,macro_roc_auc_ovo[1]]
    return(check)

# In[5]
    ##########################  Logistic Regression Classifier  ################################
def lr_fun(X_train,y_train,X_test,y_test):

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    LR_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Logistic Regression Accuracy:",LR_acc)
    
    LR_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Precision:",LR_prec)
    
    LR_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Recall:",LR_recall)
    
    LR_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression F1 weighted:",LR_f1_weighted)
    
    LR_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Logistic Regression F1 macro:",LR_f1_macro)
    
    LR_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Logistic Regression F1 micro:",LR_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix LR : \n", confuse)
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [LR_acc,LR_prec,LR_recall,LR_f1_weighted,LR_f1_macro,LR_f1_micro,macro_roc_auc_ovo[1]]
    return(check)




# In[4]

pca2 = PCA(n_components=20)
#pca2.fit(dataset_preprocessed[1:rows_number,:])
#x_3d = pca2.transform(dataset_preprocessed[1:rows_number,:])
pca2.fit(final_orig_data)
pca_final_mat = pca2.transform(temp_matrix)

##Train-Test Split
#X_train, X_test, y_train, y_test = train_test_split(pca_final_mat, true_label, 
#                                                    test_size=0.3, random_state=0)


# In[5]
# Lasso Dimensionality Reduction Logic Starts Here ########

X_train, X_test, y_train, y_test = train_test_split(
    pd.DataFrame(final_orig_data),true_label,
    test_size=0.3,
    random_state=0)
X_train.shape, X_test.shape

scaler = StandardScaler()
scaler.fit(X_train.fillna(0))

sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
sel_.fit(scaler.transform(X_train.fillna(0)), y_train)

sel_.get_support()

selected_feat = X_train.columns[(sel_.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
      np.sum(sel_.estimator_.coef_ == 0)))

#X_train_selected = sel_.transform(X_train.fillna(0))
#X_test_selected = sel_.transform(X_test.fillna(0))
#X_train_selected.shape, X_test_selected.shape
#
#X_train = X_train_selected
#X_test = X_test_selected
#
#final_lasso_data = np.concatenate((np.array(X_train),np.array(X_test)), axis=0)
#final_lasso_labels = np.concatenate((np.array(y_train),np.array(y_test)), axis=0)


# Lasso Dimensionality Reduction Logic ends Here ########


# In[5]
#    for PCA
data_selected = pca_final_mat

#for Lasso
#data_selected = sel_.transform(pd.DataFrame(feature_matrix).fillna(0))

data_labels = true_label3

X = data_selected
y =  np.array(data_labels)

rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=None)
# X is the feature set and y is the target
counter = 1
# print("Accuracy   Precision   Recall   F1 (weighted)   F1 (Macro)   F1 (Micro)   ROC AUC")
svm_table = []
gauu_nb_table = []
mlp_table = []
knn_table = []
rf_table = []
lr_table = []


for train_index, test_index in rkf.split(X):
#      print("Train:", train_index, "Validation:", test_index)
#      print("Counter = ", counter)
     counter = counter+1
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
    
     svm_return = svm_fun(X_train,y_train,X_test,y_test)
     gauu_nb_return = gaus_nb_fun(X_train,y_train,X_test,y_test)
     mlp_return = mlp_fun(X_train,y_train,X_test,y_test)
     knn_return = knn_fun(X_train,y_train,X_test,y_test)
     rf_return = rf_fun(X_train,y_train,X_test,y_test)
     lr_return = lr_fun(X_train,y_train,X_test,y_test)
#    
     svm_table.append(svm_return)
     gauu_nb_table.append(gauu_nb_return)
     mlp_table.append(mlp_return)
     knn_table.append(knn_return)
     rf_table.append(rf_return)
     lr_table.append(lr_return)
#     
svm_table_final = DataFrame(svm_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC (Macro)"])
gauu_nb_table_final = DataFrame(gauu_nb_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC (Macro)"])
mlp_table_final = DataFrame(mlp_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC (Macro)"])
knn_table_final = DataFrame(knn_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC (Macro)"])
rf_table_final = DataFrame(rf_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC (Macro)"])
lr_table_final = DataFrame(lr_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC (Macro)"])


# In[5]
print(svm_table_final)
# print(gauu_nb_table_final)
# print(mlp_table_final)
# print(knn_table_final)
# print(rf_table_final)
# print(lr_table_final)

# In[5]
#taking average of all k-fold performance values
final_mean_mat = []

final_mean_mat.append(np.transpose((list(svm_table_final.mean()))))
final_mean_mat.append(np.transpose((list(gauu_nb_table_final.mean()))))
final_mean_mat.append(np.transpose((list(mlp_table_final.mean()))))
final_mean_mat.append(np.transpose((list(knn_table_final.mean()))))
final_mean_mat.append(np.transpose((list(rf_table_final.mean()))))
final_mean_mat.append(np.transpose((list(lr_table_final.mean()))))


# In[5]
final_avg_mat = DataFrame(final_mean_mat,columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC (Macro)"], 
                          index=["SVM","NB","MLP","KNN","RF","LR"])

print(final_avg_mat.to_string())
# In[5]

#taking variance of all k-fold performance values
final_var_mat = []

final_var_mat.append(np.transpose((list(svm_table_final.var()))))
final_var_mat.append(np.transpose((list(gauu_nb_table_final.var()))))
final_var_mat.append(np.transpose((list(mlp_table_final.var()))))
final_var_mat.append(np.transpose((list(knn_table_final.var()))))
final_var_mat.append(np.transpose((list(rf_table_final.var()))))
final_var_mat.append(np.transpose((list(lr_table_final.var()))))

    
 
# In[5]
final_var_mat = DataFrame(final_var_mat,columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC (Macro)"], 
                          index=["SVM","NB","MLP","KNN","RF","LR"])
print(final_var_mat.to_string())

   
# In[5]
#gnb = GaussianNB()
#y_pred = gnb.fit(X_train, y_train).predict(X_test)
#accuracy = (((y_test != y_pred).sum())/(X_test.shape[0]))*100
#print("Accuracy = ",accuracy)
#target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5'
#                , 'class 6', 'class 7', 'class 8', 'class 9', 'class 10', 'class 11'
#                , 'class 12', 'class 13', 'class 14']
#print(classification_report(y_test, y_pred, target_names=target_names))
#
#
#neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(X_train, y_train)
#y_pred_knn = neigh.predict(X_test)
#accuracy_knn = (((y_test != y_pred_knn).sum())/(X_test.shape[0]))*100
#print("Accuracy = ",accuracy_knn)
#print(classification_report(y_test, y_pred_knn, target_names=target_names))
#      
#      
#
#clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
#clf.predict_proba(X_test[:1])
#y_pred_per = clf.predict(X_test)
#accuracy_per = (((y_test != y_pred_per).sum())/(X_test.shape[0]))*100
#print("Accuracy = ",accuracy_per)
#print(classification_report(y_test, y_pred_per, target_names=target_names))
#      
#      
#
#clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
#clf.fit(X_train, y_train)
#Pipeline(steps=[('standardscaler', StandardScaler()),
#                ('svc', SVC(gamma='auto'))])
#y_pred_svm = clf.predict(X_test)
#accuracy_svm = (((y_test != y_pred_svm).sum())/(X_test.shape[0]))*100
#print("Accuracy = ",accuracy_svm)
#print(classification_report(y_test, y_pred_svm, target_names=target_names))


