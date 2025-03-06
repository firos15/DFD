import cv2
import os
from os.path import isfile, join
from os import listdir
path1='Dataset/Drug/'
realfiles1 = sorted([ f for f in listdir(path1) if isfile(join(path1,f)) ])
print(realfiles1)
data1=[]
data2=[]
labels=[]
###########################################################################
for i in realfiles1:
  try:
    path1='Dataset/Drug/'+i
    img1=cv2.imread(path1)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img1)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    cv2.imwrite('Dataset/sal/Drug/'+str(i)+'.jpg',saliencyMap)
    print("Hello")
  except:
    print("Hi")
#############################################################################
#############################################################################
#############################################################################
import cv2
import os
from os.path import isfile, join
from os import listdir
path1='Dataset/Normal/'
realfiles1 = sorted([ f for f in listdir(path1) if isfile(join(path1,f)) ])
print(realfiles1)
for i in realfiles1:
  try:
    path1='Dataset/Normal/'+i
    img1=cv2.imread(path1)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img1)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    cv2.imwrite('Dataset/sal/Normal/'+str(i)+'.jpg',saliencyMap)
    print("Hello")
  except:
    print("Hi")