# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 20:39:36 2021

@author: Sarwan
"""

# evaluate svd with logistic regression algorithm for classification
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

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

np.set_printoptions(suppress=True)
dataset_preprocessed = temp_matrix




# Visualize
pca = PCA()
pca.fit_transform(dataset_preprocessed[1:100,:])
pca_variance = pca.explained_variance_

plt.figure(figsize=(8, 6))
plt.plot(pca_variance,label='individual variance')
plt.legend()
plt.ylabel('Variance ratio')
plt.xlabel('Principal components')
plt.show()


rows_number = 100
train_data = pd.read_csv (r'D:/University/Ph.D/Advanced Machine Learning/Project/Code/train.csv')

train_data2 = train_data.to_numpy()
#dataset_path = 'G:/Train Data/'
#onlyfiles = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
onlyfiles_names = onlyfiles[1:rows_number]

onlyfiles_names2 = onlyfiles_names
for i in range(len(onlyfiles_names)):
    print("i = ",i)
    onlyfiles_names2[i] = onlyfiles_names[i].replace('.dicom','')

true_label = onlyfiles_names
for i in range(len(onlyfiles_names)):
    print("i = ",i)
    check_temp = np.where(onlyfiles_names2[i] == train_data2[:,0])
    check_temp2 = np.asarray(check_temp)
    true_label[i] = train_data2[check_temp2[0,0],2]

true_label3 = true_label

pca2 = PCA(n_components=20)
pca2.fit(dataset_preprocessed[1:rows_number,:])
x_3d = pca2.transform(dataset_preprocessed[1:rows_number,:])


#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(x_3d, true_label, 
                                                    test_size=0.3, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
accuracy = (((y_test != y_pred).sum())/(X_test.shape[0]))*100
print("Accuracy = ",accuracy)
target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5'
                , 'class 6', 'class 7', 'class 8', 'class 9', 'class 10', 'class 11'
                , 'class 12', 'class 13', 'class 14']
print(classification_report(y_test, y_pred, target_names=target_names))


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred_knn = neigh.predict(X_test)
accuracy_knn = (((y_test != y_pred_knn).sum())/(X_test.shape[0]))*100
print("Accuracy = ",accuracy_knn)
print(classification_report(y_test, y_pred_knn, target_names=target_names))
      
      

clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
clf.predict_proba(X_test[:1])
y_pred_per = clf.predict(X_test)
accuracy_per = (((y_test != y_pred_per).sum())/(X_test.shape[0]))*100
print("Accuracy = ",accuracy_per)
print(classification_report(y_test, y_pred_per, target_names=target_names))
      
      

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('svc', SVC(gamma='auto'))])
y_pred_svm = clf.predict(X_test)
accuracy_svm = (((y_test != y_pred_svm).sum())/(X_test.shape[0]))*100
print("Accuracy = ",accuracy_svm)
print(classification_report(y_test, y_pred_svm, target_names=target_names))





