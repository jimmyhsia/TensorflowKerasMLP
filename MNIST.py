import numpy as np
import pandas as pd
from keras.utils import np_utils
np.random.seed(10)
from keras.datasets import mnist


(X_train_image,y_train_label), \
(X_test_image,y_test_label)=mnist.load_data()

print('train data=',len(X_train_image))
print('test data=',len(X_test_image))


import matplotlib.pyplot as plt
def plot_image(image):
    fig=plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image,cmap='binary')
    plt.show()
plot_image(X_train_image[1])

import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,
                                  prediction,idx,num=10):
    fig=plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title= "label=" +str(labels[idx])
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
        plt.show()



plot_images_labels_prediction(X_train_image,y_train_label,[],0,10)

print('x_train_images:',X_train_image.shape)
print('x_train_images:',y_train_label.shape)

x_train = X_train_image.reshape(60000, 784).astype('float32')
x_test = X_test_image.reshape(10000, 784).astype('float32')
X_train_image[0]
x_train_normalize = x_train / 255
x_test_normalize = x_test /255
x_train_normalize[0]
y_train_label[:5]
y_trainOneHot = np_utils.to_categorical(y_train_label)
y_testOneHot = np_utils.to_categorical(y_test_label)
y_trainOneHot[:5]
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add((Dense(units=256,input_dim=784,kernel_initializer='normal',activation='relu')))
model.add((Dense(units=10,kernel_initializer='normal',activation='softmax')))
print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_train_normalize,y=y_trainOneHot,validation_split=0.2,epochs=10,batch_size=200,verbose=2)

import matplotlib.pyplot as plt
def show_train_histroy(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

show_train_histroy(train_history,'acc','val_acc')
show_train_histroy(train_history,'loss','val_loss')
scroes = model.evaluate(x_test_normalize,y_testOneHot)

print('accuracy=',scroes[1])
prediction=model.predict_classes(x_test)
prediction

plot_images_labels_prediction(X_test_image,y_test_label,prediction,idx=340)

