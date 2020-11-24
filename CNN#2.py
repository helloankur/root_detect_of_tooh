import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib.image import imread
import os
import matplotlib.pyplot as plt
import random
import cv2
import random
from tensorflow.keras import backend as k
import pickle

data_dir = 'data'

CATEGORIES = (os.listdir(data_dir))
# print(CATEGORIES)


training_data = []

IMG_SIZE = 130


def data_create():
    for category in CATEGORIES:
        path = os.path.join(data_dir, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


data_create()

X = []
y = []

for feature, label in training_data:
    X.append(feature)
    y.append(label)

# print(X)
# print(y)

y = np.array(y)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X = X / 255.

# print(y.shape)

# print(X[0])
# print(X.shape)

print(X.shape[1:])

from sklearn.model_selection import train_test_split

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from tensorflow.keras.utils import to_categorical

# print(len(X_train))
# print(len(y_train))

# print(len(X_test))
# print(len(y_test))

y = to_categorical(y)
# print(y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Building the model
model = Sequential()

# 3 convolutional layers
model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2 hidden layers
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))

# The output layer with 2 neurons, for 2 classes
model.add(Dense(2))
model.add(Activation("softmax"))

# Compiling the model using some basic parameters

from tensorflow.keras.optimizers import RMSprop

# model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['acc'])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Training the model, with 40 iterations
# validation_split corresponds to the percentage of images used for the validation phase compared to all the images
# from sklearn.model_selection import train_test_split

history = model.fit(X_train, y_train, batch_size=100, epochs=20, validation_data=(X_test, y_test))

print("Saving model to disk")
model.save('CNN_model.h5')

# Printing a graph showing the accuracy changes during the training phase

print(history.history.keys())
losses = pd.DataFrame(model.history.history)
print(losses)
losses[['loss', 'val_loss', 'accuracy', 'val_accuracy']].plot()

plt.title("Model Performance")
plt.show()
