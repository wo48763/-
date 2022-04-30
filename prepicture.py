import json
import os
from re import S
import cv2
import numpy as np
from numpy import expand_dims
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import shutil


def prepare(rowfile, picfile, jfile, savename, size, small, span):
    with open(jfile, 'r', encoding='utf8') as jFile:
        jdata = json.load(jFile)

    for i in os.listdir(rowfile):
        img = cv2.imread(os.path.join(rowfile,i))

        #調整大小、儲存
        newimg = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)
        label = i[0]
        cv2.imwrite(os.path.join(picfile,f"{i[0]}.{jdata[label]}.jpg"),newimg)
        jdata[label] += 1
        print(label,"已讀取",jdata[label],"張了")

    with open(jfile, 'w', encoding='utf8') as jFile:
        json.dump(jdata,jFile,indent = 4)

    print(jdata)

    datagen = ImageDataGenerator(   rotation_range=60, # 角度值，0~180，影象旋轉
                                        width_shift_range=0.1, # 水平平移，相對總寬度的比例
                                        height_shift_range=0.1, # 垂直平移，相對總高度的比例
                                        shear_range=0.4, # 隨機錯切換角度
                                        zoom_range=0.1, # 隨機縮放範圍
                                        horizontal_flip=True, # 一半影象水平翻轉
                                        fill_mode='nearest' # 填充新建立畫
                                        )
    def gen(img,num):
        imgp = img_to_array(img)
        samples = expand_dims(imgp,0)
        it = datagen.flow(samples, batch_size=1)
        ap = [it.next()[0].astype('uint8') for i in range(num)]
        return ap

    spict = dict(jdata)

    tempfile = picfile+"\\..\\temp048596"
    if os.path.isdir(tempfile):
        shutil.rmtree(tempfile)
    os.mkdir(tempfile)
        
    for n in jdata.keys():
        for i in range(jdata[n]):
            need = (span - spict[n]) // (jdata[n]-i)
            img = cv2.imread(os.path.join(picfile, f"{n}.{i}.jpg"))
            g = gen(img,need)
            cv2.imwrite(f"{tempfile}/{n}.{i}.jpg", cv2.resize(img, (small,small),interpolation=cv2.INTER_AREA))
            for p in g:
                cv2.imwrite(f"{tempfile}/{n}.{spict[n]}.jpg", cv2.resize(p, (small,small),interpolation=cv2.INTER_AREA))
                spict[n] += 1
        print(f"{n}span完成")
    
    #存成pickle
    lis = os.listdir(tempfile)
    dt = {}
    dt["data"] = np.empty((len(lis),small,small,3),dtype="float")
    dt["label"] = np.empty((len(lis)),dtype=int)

    for i in range(len(lis)):
        dt["data"][i] = cv2.imread(os.path.join(tempfile,lis[i]))
        dt["label"][i] = int(lis[i][0])

    import random

    index = list(range(len(lis)))
    random.shuffle(index)
    dt["data"] = dt["data"][index]
    dt["label"] = dt["label"][index]

    dt["data"] = dt["data"].astype("float32")/255.0
    dt["label"] = dt["label"]


    import pickle
    with open(savename,'wb') as f:
        pickle.dump(dt,f)
        

prepare(
    rowfile = r".\picture",
    picfile = r".\trainpic",
    jfile = r".\d_num.json",
    savename = r".\train.pickle",
    size = 200,
    small = 32,
    span = 2000
)

prepare(rowfile= r".\test_picture",
    picfile= r".\testpic",
    jfile= r".\t_num.json",
    savename= r".\test.pickle",
    size= 200,
    small= 32,
    span= 100)