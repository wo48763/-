import keras
import tensorflow 
import numpy as np
import sklearn
from sklearn import metrics
import cv2
import matplotlib.pyplot as plt

filename = r".jpg"
model = keras.models.load_model(r".\killqueen.h5")

img = cv2.imread(filename)
simg = cv2.resize(img,(32,32),interpolation=cv2.INTER_AREA).astype("float32")

simg = simg/255.0

simg = np.reshape(simg,(1,32,32,3))

pre = model.predict(simg)
pre = np.argmax(pre, axis=1)[0]

cv2.imshow(f"{pre}", cv2.resize(img,(img.shape[1]//6,img.shape[0]//6),interpolation=cv2.INTER_AREA))
cv2.waitKey(0)