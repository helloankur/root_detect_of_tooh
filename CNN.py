import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import shuffle
from tensorflow.keras.models import Sequential
from matplotlib.image import imread

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(130, 130, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))


model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['acc'])
#model.compile(loss='binary_crossentropy',optimizer="adam",metrics=["accuracy"])

image_gen = ImageDataGenerator(rotation_range=30,
                               width_shift_range=0.5,
                               height_shift_range=0.5,
                               rescale=1. / 255.,
                               shear_range=0.1,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               vertical_flip=True,
                               fill_mode='nearest', validation_split=0.3)

train_data_gen = image_gen.flow_from_directory('data',
                                               target_size=(130, 130),
                                               batch_size=128,
                                               class_mode='binary',
                                               subset="training",
                                               # save_to_dir='split\\train',
                                               # save_format="jpeg"
                                               )
print(train_data_gen.class_indices)

val_data_gen = image_gen.flow_from_directory('data', target_size=(130, 130),
                                             batch_size=128,
                                             class_mode='binary',
                                             subset="validation", shuffle=False,
                                             # save_to_dir='split\\validation',
                                             # save_format="jpeg"
                                             )
print(val_data_gen.class_indices)

from datetime import datetime
from tensorflow import keras

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

history = model.fit(train_data_gen,
                    steps_per_epoch=8,
                    epochs=200,
                    verbose=1,
                    validation_data=val_data_gen,
                    callbacks=[tensorboard_callback]
                    )

model.save('tooth_detect_update.h5')

print(history.history.keys())

losses = pd.DataFrame(model.history.history)

print(losses)
#losses[['loss', 'val_loss', 'accuracy', 'val_accuracy']].plot()
losses[['loss', 'val_loss', 'acc', 'val_acc']].plot()
plt.title("Model Performance")
plt.show()


