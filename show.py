import keras
import tensorflow 
import numpy as np
import sklearn
from sklearn import metrics
import cv2
import matplotlib.pyplot as plt

def loaddata(file):
    import pickle
    with open(file, 'rb')as f:
        dt = pickle.load(f, encoding="bytes")
    return (dt["data"],dt["label"])

model = keras.models.load_model(r".\killqueen.h5")
(x_test,y_test) = loaddata(r".\test.pickle")

pre = model.predict(x_test)
pre = np.argmax(pre,axis=1)

print(metrics.accuracy_score(y_test,pre))

import pandas as pd
ht = pd.crosstab(np.array(y_test).reshape(-1),pre)
print(ht)
cfu_me = metrics.confusion_matrix(y_test,pre)
print(cfu_me.shape)
cfu_me = cfu_me.astype('float')/cfu_me.sum(axis=1)
print(cfu_me.sum(axis=0))
plt.figure(figsize=(cfu_me.shape[0],cfu_me.shape[1]))
plt.imshow(cfu_me,cmap='Blues')
plt.colorbar()
plt.xticks(list(range(cfu_me.shape[0])))
plt.yticks(list(range(cfu_me.shape[0]))) 
thresh = cfu_me.max()/2.0
shp = cfu_me.shape
for i in range(shp[0]):
    for j in range(shp[1]):
        plt.text(j,i,f"{cfu_me[i,j]*100:2.0f}%",horizontalalignment="center",color="white" if cfu_me[i,j] > thresh else "black")
        
plt.ylabel("True")
plt.xlabel("predicted")
plt.tight_layout()
plt.show()

