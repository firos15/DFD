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
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
from tensorflow.keras.layers import Dropout
height=227
width=227
data1=[]
labels=[]
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
print("data1:",len(data1))
print("labels:",len(labels))
data1=np.array(data1)
labels=np.array(labels)
print(len(labels))
input_shape=(227, 227,3)
seq=Sequential()
seq.add(Conv2D(96, kernel_size=(11,11), strides= 4,padding= 'valid', activation= 'relu',
                      input_shape= input_shape,
                      kernel_initializer= 'he_normal'))
seq.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                      padding= 'valid', data_format= None))

seq.add(Conv2D(256, kernel_size=(5,5), strides= 1,
                padding= 'same', activation= 'relu',
                kernel_initializer= 'he_normal'))
seq.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                      padding= 'valid', data_format= None)) 
seq.add(Conv2D(384, kernel_size=(3,3), strides= 1,
                padding= 'same', activation= 'relu',
                kernel_initializer= 'he_normal'))

seq.add(Conv2D(384, kernel_size=(3,3), strides= 1,
                padding= 'same', activation= 'relu',
                kernel_initializer= 'he_normal'))

seq.add(Conv2D(256, kernel_size=(3,3), strides= 1,
                padding= 'same', activation= 'relu',
                kernel_initializer= 'he_normal'))

seq.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                      padding= 'valid', data_format= None))

seq.add(Flatten())
seq.add(Dense(4096, activation= 'relu'))
seq.add(Dropout(0.5))
seq.add(Dense(1000, activation= 'relu'))
seq.add(Dense(1,activation='sigmoid'))
seq.compile(optimizer= tf.keras.optimizers.Adam(0.001),loss='binary_crossentropy',metrics=['accuracy'])
checkpoint = ModelCheckpoint(
    'Model/s1.h5',
    monitor='accuracy',
    save_best_only=True,
    verbose=1
)
seq.fit( 
  data1,
  labels,
  epochs=20,
  callbacks=[checkpoint])
#######################################################################################################
#######################################################################################################
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
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
from tensorflow.keras.layers import Dropout
height=227
width=227
data1=[]
labels=[]
###########################################################################
                          #Image Reading
###########################################################################
path1='Dataset/sal/Drug/'
realfiles1 = sorted([ f for f in listdir(path1) if isfile(join(path1,f)) ])
print(realfiles1)
for i in realfiles1:
  try:
    path1='Dataset/sal/Drug/'+i
    img=cv2.imread(path1)
    img=cv2.resize(img,(height,width))
    data1.append(img)
    labels.append(0)
  except:
    pass
##############################################################################
##############################################################################
path1='Dataset/sal/Normal/'
realfiles1 = sorted([ f for f in listdir(path1) if isfile(join(path1,f)) ])
print(realfiles1)
for i in realfiles1:
  try:
    path1='Dataset/sal/Normal/'+i
    img=cv2.imread(path1)
    img=cv2.resize(img,(height,width))
    data1.append(img)
    labels.append(1)
  except:
    pass
##############################################################################
print("data1:",len(data1))
print("labels:",len(labels))
data1=np.array(data1)
labels=np.array(labels)
print(len(labels))
input_shape=(227, 227,3)
seq=Sequential()
seq.add(Conv2D(96, kernel_size=(11,11), strides= 4,padding= 'valid', activation= 'relu',
                      input_shape= input_shape,
                      kernel_initializer= 'he_normal'))
seq.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                      padding= 'valid', data_format= None))

seq.add(Conv2D(256, kernel_size=(5,5), strides= 1,
                padding= 'same', activation= 'relu',
                kernel_initializer= 'he_normal'))
seq.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                      padding= 'valid', data_format= None)) 
seq.add(Conv2D(384, kernel_size=(3,3), strides= 1,
                padding= 'same', activation= 'relu',
                kernel_initializer= 'he_normal'))

seq.add(Conv2D(384, kernel_size=(3,3), strides= 1,
                padding= 'same', activation= 'relu',
                kernel_initializer= 'he_normal'))

seq.add(Conv2D(256, kernel_size=(3,3), strides= 1,
                padding= 'same', activation= 'relu',
                kernel_initializer= 'he_normal'))

seq.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                      padding= 'valid', data_format= None))

seq.add(Flatten())
seq.add(Dense(4096, activation= 'relu'))
seq.add(Dropout(0.5))
seq.add(Dense(1000, activation= 'relu'))
seq.add(Dense(1,activation='sigmoid'))
seq.compile(optimizer= tf.keras.optimizers.Adam(0.001),loss='binary_crossentropy',metrics=['accuracy'])
checkpoint = ModelCheckpoint(
    'Model/s2.h5',
    monitor='accuracy',
    save_best_only=True,
    verbose=1
)
seq.fit( 
  data1,
  labels,
  epochs=20,
  callbacks=[checkpoint])