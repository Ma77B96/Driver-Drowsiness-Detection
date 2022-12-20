import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

DATA_DIR_OPEN = "./dataset_new/train/"


labels = ['Closed', 'Open']

IMG_SIZE = 64

data = []

for label in labels:
    path = os.path.join(DATA_DIR_OPEN, label)
    class_num = labels.index(label)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            img_array = img_array / 255
            resized_aray = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
            data.append([resized_aray, class_num])
        except Exception as e:
            print(e)

print(data)


# separo labels e features

X = []
y = []

for feature, label in data:
    X.append(feature)
    y.append(label)

X,y = shuffle(X,y)


print('---------------------------', y)


# faccio il reshape dell'array

X = np.array(X)

X = X.reshape(-1,64,64,3)

y = np.array(y)


# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

print(len(X_train))
print(len(X_test))



# model

model = Sequential()

model.add(Conv2D(256, (3, 3), activation="relu", input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")

model.summary()



model.fit(X_train, y_train, epochs=10)

# valuto il modello 

metrics_train = model.evaluate(X_train, y_train, verbose=0)
metrics_test = model.evaluate(X_test, y_test, verbose=0)

print("Train Accuracy = %.4f - Train Loss = %.4f" % (metrics_train[1], metrics_train[0]))
print("Test Accuracy = %.4f - Test Loss = %.4f" % (metrics_test[1], metrics_test[0]))

# salvo il modello

model.save("drowiness_new_color.h5")

prediction = model.predict(X_test)
print(prediction)