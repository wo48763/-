import pickle
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

batch_size = 64
num_classes = 10
Epochs = 30
size = 32

def loaddata(file):
    import pickle
    with open(file, 'rb')as f:
        dt = pickle.load(f, encoding="bytes")
    return (dt["data"],dt["label"])

(x_train,y_train) = loaddata(r".\train.pickle") 
(x_test,y_test) = loaddata(r".\test.pickle")

print(y_train)

print(len(x_train),"個train")
print(x_train[0].shape)
y_train_OneHot = tensorflow.keras.utils.to_categorical(y_train)
print(np.unique(y_train))
y_test_OneHot = tensorflow.keras.utils.to_categorical(y_test)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(size,size,3), activation='relu', padding='same'))
model.add(Dropout(rate=0.3))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(3,3), input_shape=(size,size,3), activation='relu', padding='same'))
model.add(Dropout(rate=0.3))
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Flatten())


model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.34))
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.34))



model.add(Dense(num_classes, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

train_history = model.fit(
    x_train,y_train_OneHot,
    validation_split=0.2 ,
    epochs=Epochs,
    batch_size=batch_size,
    verbose=1,
    shuffle=False
)
    
import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy' if train_acc == 'acc' else 'Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
show_train_history('acc','val_acc')
show_train_history('loss','val_loss')


# 模型評估
score = model.evaluate(x_test, y_test_OneHot, verbose=1)
pre = model.predict(x_test)
pre = np.argmax(pre,axis=1)

print("train accuracy:", f"{train_history.history['acc'][-1]:.6f}")
print("test  accuracy:", f"{score[1]:.6f}")
print('train loss    :', f"{train_history.history['loss'][-1]:.6f}")
print('Test loss     :', f"{score[0]:.6f}")

print("test precision:", f"{precision_score(y_test,pre,average='macro',zero_division=0):.6f}")
print("recall        :", f"{recall_score(y_test,pre,average='macro'):.6f}")
print("f-1 score     :", f"{f1_score(y_test,pre,average='macro'):.6f}")

# 儲存模型
try:
    model.save(r".\killqueen.h5")
    print("success")
except:
    print("error")