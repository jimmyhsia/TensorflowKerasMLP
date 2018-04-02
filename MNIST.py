
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
from keras.utils import np_utils
np.random.seed(10)


# In[48]:


from keras.datasets import mnist


# In[49]:


(X_train_image,y_train_label), (X_test_image,y_test_label)=mnist.load_data()


# In[50]:


print('train data=',len(X_train_image))
print('test data=',len(X_test_image))


# In[51]:


import matplotlib.pyplot as plt
def plot_image(image):
    fig=plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image,cmap='binary')
    plt.show()


# In[52]:


plot_image(X_train_image[1])


# In[53]:


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


# In[54]:


plot_images_labels_prediction(X_train_image,y_train_label,[],0,10)


# In[55]:


print('x_train_images:',X_train_image.shape)
print('x_train_images:',y_train_label.shape)


# In[56]:


x_train = X_train_image.reshape(60000, 784).astype('float32')
x_test = X_test_image.reshape(10000, 784).astype('float32')


# In[57]:


X_train_image[0]


# In[58]:


x_train_normalize = x_train / 255
x_test_normalize = x_test /255


# In[59]:


x_train_normalize[0]


# In[60]:


y_train_label[:5]


# In[61]:


y_trainOneHot = np_utils.to_categorical(y_train_label)
y_testOneHot = np_utils.to_categorical(y_test_label)


# In[62]:


y_trainOneHot[:5]


# In[63]:


from keras.models import Sequential
from keras.layers import Dense


# In[64]:


model = Sequential()


# In[65]:


model.add((Dense(units=256,input_dim=784,kernel_initializer='normal',activation='relu')))


# In[66]:


model.add((Dense(units=10,kernel_initializer='normal',activation='softmax')))


# In[67]:


print(model.summary())


# In[68]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[69]:


train_history=model.fit(x=x_train_normalize,y=y_trainOneHot,validation_split=0.2,epochs=10,batch_size=200,verbose=2)


# In[70]:


import matplotlib.pyplot as plt
def show_train_histroy(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()


# In[71]:


show_train_histroy(train_history,'acc','val_acc')


# In[72]:


show_train_histroy(train_history,'loss','val_loss')


# In[73]:


scroes = model.evaluate(x_test_normalize,y_testOneHot)


# In[75]:


print()
print('accuracy=',scroes[1])


# In[79]:


prediction=model.predict_classes(x_test)


# In[80]:


prediction


# In[82]:


plot_images_labels_prediction(X_test_image,y_test_label,prediction,idx=340)

