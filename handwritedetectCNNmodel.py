# -*- coding: utf-8 -*-

# *Import relative packages and modules*
import os # os module (for loading data from datapath)
import numpy as np # math operation for multi-dimensional arrays and matrices
from PIL import Image # PIL has image handling and processing modules
from keras.models import Sequential # for building simple sequential training model
# Sequential就是一層一層往下，沒有分叉(if)，也沒有迴圈(loop)，標準一層一層傳遞的神經網路叫Sequential
# for building different training layer including CNN convolutional layer and pooling layer
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from keras.utils import np_utils # for transforming data labels to one-hot-encoding
# create check point for detecting and saving the improving weights of the training model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt # for drawing figures and charts
import pandas as pd # for caculating and showing the Confusion Matrix of the predictions

# *Data preprocessing*
resize = 28; # set the image size to be 28*28
# this two global variables are used for storing the original label data
data_train_y_org=[] # store training data label
data_test_y_org = [] # store testing data label

# important!!!
# standardization(self-defined) to detect,modify and avoid the outliers
def self_defined_standardization(image):
    temp_image = (np.array(image)/255) # operate normalization to the image data
    temp_mean = np.mean(temp_image) # calculate the average value of elements in numpy array
    # if the element > the average value then change it to 1, else change it to 0
    temp_image = np.where(temp_image > temp_mean, 1, 0)
    return temp_image

# data preprocessing function
def data_x_y_preprocess(datapath):
    row , col = 28,28 # define the size of image
    data_x = np.zeros((28,28)).reshape(1,28,28) # for storing image data (the first data is numpy.zeros array)
    data_y = [] # record label data of images
    picturecount = 0 # record the amount of image
    classes = 10 # there are 10 categories of digit number 
    # read all image files in the directory folder at the datapath
    for root,dirs,files in os.walk(datapath):   
        for f in files:
            label = int(root.split("\\")[9]) # get the number label in the datapath string
            data_y.append(label) # store the label in data_y
            fullpath = os.path.join(root,f) # combine dirpath and filename, get the datapath of file
            img = Image.open(fullpath) # open the image
            # standardization(self-defined) to detect,modify and avoid the outliers
            std_img = self_defined_standardization(img)
            img=(1-std_img).reshape(1,28,28) # operating normalization and reshape on the image
            data_x=np.vstack((data_x,img)) # store the image data in data_x
            picturecount += 1 # the record the amount of image plus 1
    data_x = np.delete(data_x,[0],0) # delete the first declared numpy.zeros array
    # modify the data format (the amount of image, image_row, image_col, color channel=1(gray scale) )
    data_x = data_x.reshape(picturecount,row,col,1)
    # keep the original label data, 供 cross tab function 使用 # use global variables to record
    if (datapath.split("\\")[7]=="train_image"):
        global data_train_y_org # state that the using variable is global variable
        data_train_y_org = data_y # store the training data label to data_train_y_org
    elif (datapath.split("\\")[7]=="test_image"):
        global data_test_y_org # state that the using variable is global variable
        data_test_y_org = data_y # store the testing data label to data_test_y_org
    data_y = np_utils.to_categorical(data_y,classes) # transform the label data to one-hot-encoding
    return data_x,data_y

# input training dataset
[data_train_x,data_train_y] = data_x_y_preprocess("PATH\\train_image\\")
# input testing dataset
[data_test_x,data_test_y] = data_x_y_preprocess("PATH\\test_image\\")

# transform label list to ndarray
data_train_y_label = np.array(data_train_y_org)
data_test_y_label = np.array(data_test_y_org)

# show the data type
print('data_train_x_image:',data_train_x.shape) # (2450, 28, 28, 1)
print('data_train_y_label',data_train_y_label.shape) # (2450,)

# show the amount of the data
print ('train data = ',len(data_train_x)) # 2450
print ('test data = ',len(data_test_x)) # 1700

# function for drawing the image with its actual label and prediction
def plot_image_label_prediction(images,labels,prediction,idx,idxinc,num=10):
    # the parameter "num" decides the amount of images showing
    fig = plt.gcf() # get the current figure
    fig.set_size_inches(12,14) # set the size of the figure
    if num > 25: num=25 # enforce that the maximum amount of the subplot figure is 25 
    for i in range (0,num):        
        ax = plt.subplot(5,5,1+i) # subplot(nrows, ncols, index) # the index of subplot(start from 1)
        ax.imshow(images[idx].reshape(28,28), cmap='binary') # show the image figure of subplot, and represent in gray scale
        title = "label=" +str(labels[idx]) # set subplot title to be the label of the image
        if len(prediction)>0: # if prediction parameter is valid
            title += ",predict=" + str(prediction[idx]) # then add the image prediction result to the subplot title
        ax.set_title (title,fontsize=10) # set subplot title and the font size of it
        idx += idxinc # jump to the next "idxinc(amount)" data
    plt.show() # show the whole figure

plot_image_label_prediction(data_train_x,data_train_y_label,[],0,5,10) # show first part of the image data

# *Build the CNN training model*
# build simple sequential training model
model = Sequential()
# convolutional layer # 建立卷積層，filter=32，即output space深度，Kernel Size=3*3,activation function採用relu #第一層須設input shape
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), input_shape=(resize, resize, 1), padding='same', activation='relu'))
# normalization layer
model.add(BatchNormalization())
# pooling layer #建立池化層，池化大小2*2，取最大值
model.add(MaxPooling2D(pool_size=(2,2),strides=None))
# convolutional layer # 建立卷積層，filter=64，即output size，Kernel Size=3*3,activation function採用relu
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
# normalization layer
model.add(BatchNormalization())
# pooling layer #建立池化層，池化大小2*2，取最大值
model.add(MaxPooling2D(pool_size=(2,2),strides=None))
# convolutional layer # 建立卷積層，filter=128，即output size，Kernel Size=3*3,activation function採用relu
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
# pooling layer #建立池化層，池化大小2*2，取最大值
model.add(MaxPooling2D(pool_size=(2,2),strides=None))
# flatten layer (2D -> 1D) # Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡
model.add(Flatten())
# dropout layer # Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例：0.2
model.add(Dropout(0.2))
# fully connection layer # 全連接層：128個output
model.add(Dense(128,activation='relu'))
# dropout layer # Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例：0.25
model.add(Dropout(0.25))
# output layer # 使用softmax activation function，將結果分類(units=10。表示分10類)
model.add(Dense(10,activation='softmax'))

# show the summary of the model
model.summary()
# compile the training model # 編譯：選擇損失函數、優化方法及成效評量方式
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['accuracy'])

# checkpoint # 建立保存點保存驗證集上驗證結果最好的權重
filepath= "150epoch_06_weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# training the model # 進行訓練，訓練過程會存在train_history變數中
train_history = model.fit(
    data_train_x,
    data_train_y,
    # batch_size = 32, # set the batch size
    epochs=150, # set the amount of epoch
    validation_split = 0.1, # set the proportion of validation data set in training data set
    shuffle=True, # shuffle the training data before each epoch
    callbacks=callbacks_list, # set the list of callbacks to apply during training
)

# save model
model.save('CNN_model_150epoch_06.h5')

#  test the model and show the test result # 顯示損失函數、訓練成果(分數)
scores = model.evaluate(data_test_x,data_test_y)
print('accuracy=',scores[1])
print('loss=',scores[0])

# function for plotting the optimization process line chart
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train]) # plot the training process line of every epoch
    plt.plot(train_history.history[validation]) # plot the validation process line of every epoch
    plt.title('Train History') # set the figure title
    plt.ylabel(train) # set the y axis title
    plt.xlabel("Epoch") # set the x axis title
    plt.legend(['train','validation'],loc='upper right') # place legend on both axes
    plt.show() # show the figure

# plotting the optimization process line chart of accuracy and validation accuracy
show_train_history(train_history, 'accuracy', 'val_accuracy')
# plotting the optimization process line chart of loss and validation loss
show_train_history(train_history, 'loss', 'val_loss')

# predict the category of label of testing data
predictions = model.predict_classes(data_test_x)
# 計算`「混淆矩陣」(Confusion Matrix)，顯示測試集分類的正確及錯認總和數
print( pd.crosstab(data_test_y_label, predictions, rownames=['label'], colnames=['predict']) )

# show the prediction image
plot_image_label_prediction(data_test_x,data_test_y_label,predictions,0,5,25)

# function for drawing the wrong prediction image with its actual label and the wrong prediction
def plot_image_label_prediction_wrong(images,labels,prediction,idx,num=10):
    # the parameter "num" decides the amount of images showing
    fig = plt.gcf() # get the current figure
    fig.set_size_inches(12,14) # set the size of the figure
    if num > 25: num=25 # enforce that the maximum amount of the subplot figure is 25
    counter = 0 # record the amount of the image that have been plotted
    for i in range (idx,len(prediction)): # traverse all the prediction of testing data
        if (labels[i]!=prediction[i]): # if the prediction is wrong (actual image label is not equal prediction)
            # plotting the wrong predicted image
            ax = plt.subplot(5,5,1+counter) # subplot(nrows, ncols, index) # the index of subplot(start from index 1)
            ax.imshow(images[i].reshape(28,28), cmap='binary') # show the image figure of subplot, and represent in gray scale
            # set subplot title to be the actual label and the wrong prediction result of the image 
            title = "label=" +str(labels[i]) + ",predict=" + str(prediction[i]) 
            ax.set_title (title,fontsize=10) # set subplot title and the font size of it
            counter += 1 # "the amount of the image that have been plotted" plus 1
        if counter == num: break # if the amount of the image that have been plotted reach the maximum, then stop plotting
    plt.show() # show the whole figure

# show the first 25 wrong prediction image of the model
plot_image_label_prediction_wrong(data_test_x,data_test_y_label,predictions,0,25)