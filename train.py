import os
import numpy as np
from os.path import isfile, join
from os import listdir
import cv2
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
height=227
width=227
data1=[]
data2=[]
labels=[]
models1=load_model('Model/s1.h5')
models2=load_model('Model/s2.h5')
###########################################################################
                          #Image Reading
###########################################################################
path1='Dataset/Drug/'
realfiles1 = sorted([ f for f in listdir(path1) if isfile(join(path1,f)) ])
print(realfiles1)
for i in realfiles1:
  try:
    path1='Dataset/Drug/'+i
    img=cv2.imread(path1)
    img=cv2.resize(img,(height,width))
    data1.append(img)
    labels.append(0)
  except:
    pass
##############################################################################
path1='Dataset/sal/Drug'
realfiles1 = sorted([ f for f in listdir(path1) if isfile(join(path1,f)) ])
print(realfiles1)
for i in realfiles1:
  try:
    path1='Dataset/sal/Drug/'+i
    img=cv2.imread(path1)
    img=cv2.resize(img,(height,width))
    data2.append(img)
  except:
    pass
##############################################################################
##############################################################################
path1='Dataset/Normal/'
realfiles1 = sorted([ f for f in listdir(path1) if isfile(join(path1,f)) ])
print(realfiles1)
for i in realfiles1:
  try:
    path1='Dataset/Normal/'+i
    img=cv2.imread(path1)
    img=cv2.resize(img,(height,width))
    data1.append(img)
    labels.append(1)
  except:
    pass
##############################################################################
path1='Dataset/sal/Normal/'
realfiles1 = sorted([ f for f in listdir(path1) if isfile(join(path1,f)) ])
print(realfiles1)
for i in realfiles1:
  try:
    path1='Dataset/sal/Normal/'+i
    img=cv2.imread(path1)
    img=cv2.resize(img,(height,width))
    data2.append(img)
  except:
    pass
###############################################################################
###############################################################################
###############################################################################
# print("#########data1#########")
# print(data1)
# print("#########data2#########")
# print(data2)
# print("#######################")
print("data1:",len(data1))
print("labels:",len(labels))
data1=np.array(data1)
data2=np.array(data2)
labels=np.array(labels)
print(len(labels))
########################################################
########################################################
ip=[]
for i in range(0,400):
  a=data1[i]
  a=np.expand_dims(a, axis=0)
  model11=models1.predict(a)
  ###########################
  b=data2[i]
  b=np.expand_dims(b, axis=0)
  model22=models2.predict(b)
  ###########################
  combined=np.concatenate([model11,model22])
  print(combined)
  ip.append(combined)
xtrain,xtest,ytrain,ytest=train_test_split(ip,labels,test_size=0.1,random_state=42)
xtrain=np.array(xtrain)
xtest=np.array(xtest)
print("Train shape(df) :",xtrain.shape)
print("Test shape(df) :",xtest.shape)
# m1=np.stack(train['Features'].values)
xtrain = xtrain.reshape(xtrain.shape[0],-1)
# m2=np.stack(train['label'].values)
# m3=np.stack(test['Features'].values)
xtest = xtest.reshape(xtest.shape[0],-1)
# m4=np.stack(test['label'].values)
print("Train shape(df) :",xtrain.shape)
print("Test shape(df) :",xtest.shape)
#########################################
                #SVM
#########################################
from sklearn.svm import SVC
print('#####Hi######')
msvm = SVC()
msvm.fit(xtrain,ytrain)
y_pred=msvm.predict(xtest)
print('#######SVM########')
print('Accuracy: %.3f' % accuracy_score(ytest, y_pred))
print('Precision: %.3f' % precision_score(ytest, y_pred))
print('Recall: %.3f' % recall_score(ytest, y_pred))
conf_matrix = confusion_matrix(y_true=ytest, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
  for j in range(conf_matrix.shape[1]):
    ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
# pickle.dump(msvm, open('Model/svm.pkl', 'wb'))
#plt.savefig('Confusion Matrix/SVM.png')
plt.show()
##########################################
           #Random Forest
##########################################
ran = RandomForestClassifier(n_estimators = 100)
ran.fit(xtrain, ytrain)
y_pred=ran.predict(xtest)
print("#########Random Forest#########")
print('Accuracy: %.3f' % accuracy_score(ytest, y_pred))
print('Precision: %.3f' % precision_score(ytest, y_pred))
print('Recall: %.3f' % recall_score(ytest, y_pred))
conf_matrix = confusion_matrix(y_true=ytest, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
  for j in range(conf_matrix.shape[1]):
    ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
pickle.dump(ran, open('Model/ran.pkl', 'wb'))
plt.savefig('Confusion Matrix/Random Forest.png')
plt.show()
##########################################
#########################################
           #KNN
#########################################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(xtrain, ytrain)
y_pred=knn.predict(xtest)
print("#########KNN#########")
print('Accuracy: %.3f' % accuracy_score(ytest, y_pred))
print('Precision: %.3f' % precision_score(ytest, y_pred))
print('Recall: %.3f' % recall_score(ytest, y_pred))
conf_matrix = confusion_matrix(y_true=ytest, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
  for j in range(conf_matrix.shape[1]):
    ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
pickle.dump(knn, open('Model/knn.pkl', 'wb'))
plt.savefig('Confusion Matrix/KNN.png')
plt.show()
##########################################