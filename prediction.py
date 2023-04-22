import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras 
import seaborn as sns
import os

import glob
import cv2

filenames = os.listdir('data/combined')

alluvial = []
black = []
desert = []
red = []
for files in filenames:
    if files[0:8] == 'alluvial':
        alluvial.append(files)
    elif files[0:5] == 'black':
        black.append(files)
    elif files[0:6] == 'desert':
        desert.append(files)
    elif files[0:3] == 'red':
        red.append(files)
        
print(len(alluvial),len(black),len(desert),len(red))

image_directory = 'images_resized1/combined/'
image_extensions = ['png','jpg']

files = []
[files.extend(glob.glob(image_directory + '*.' + e)) for e in image_extensions]
soil_images = np.asarray([cv2.imread(file) for file in files])

labels = []
for i in range(0,len(filenames)):
    files = filenames[i]    
    if files[0:8] == 'alluvial':
        labels.append(0)
    elif files[0:5] == 'black':
        labels.append(1)
    elif files[0:6] == 'desert':
        labels.append(2)
    elif files[0:3] == 'red':
        labels.append(3)

labels.count(0), labels.count(1),labels.count(2), labels.count(3)

X = soil_images
Y = np.asarray(labels)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,shuffle=True,random_state=35,stratify=Y)
X_train_scaled = X_train/255
X_test_scaled = X_test/255

import tensorflow_hub as hub
mobile_net_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
pretrained_model = hub.KerasLayer(mobile_net_model,input_shape=(224,224,3),trainable=False)

from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout

# number_of_classes = 4

model = Sequential()

model.add(pretrained_model)

model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(4,activation='softmax'))

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['acc']
)

history = model.fit(X_train_scaled,Y_train,epochs=2,validation_split=0.25,batch_size=16)

model.save_weights('abc.h5')

def soil_predictor(img_path):
    from PIL import ImageFile
    from PIL import Image

    img_test = cv2.imread(img_path)
    img_resize = cv2.resize(img_test,(224,224))
    img_scaled = img_resize/255
    img_reshaped = np.reshape(img_scaled,[1,224,224,3])
    input_pred = model.predict(img_reshaped)
    input_label = np.argmax(input_pred)

    if input_label == 0:
        return "Alluvial Soil"
    elif input_label == 1:
        return "Black Soil"
    elif input_label == 2:
        return "Desert Soil"
    elif input_label == 3:
        return "Red Soil"
