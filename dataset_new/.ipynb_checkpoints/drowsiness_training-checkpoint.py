import sys
import os
import PIL
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model
import cv2

CUR_DIR = os.getcwd()
FIRST_TRAIN_FOLDER = "dataset_new/train/Closed/"
SECOND_TRAIN_FOLDER = "../../train/Open/"
FIRST_TEST_FOLDER = "dataset_new/test/Closed/"
SECOND_TEST_FOLDER = "dataset_new/test/Open/"


x_train = []
y_train = []


for i in os.listdir(os.getcwd()):
    try:
        im = cv2.imread(i)
        image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        image = image.resize((24,24))
        image = np.asarray(image)
        x_train.append(image)
        y_train.append("0")
    except:
        print("Error loading image")

for i in os.listdir(SECOND_TRAIN_FOLDER):
    try:
        im = cv2.imread(i)
        image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        image = image.resize((24,24))
        image = np.asarray(image)
        x_train.append(image)
        y_train.append("1")
    except:
        print("Error loading image")


x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train)
print(y_train)


model = Sequential()
model.add(Conv2D(input_shape=(30,30,1), filters=32,kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train)











