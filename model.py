import keras
from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from keras.preprocessing.image import ImageDataGenerator
from Data_Genetator import Data_Generator


# Image preprocessing
img_h = 34
img_w = 152
dirpath = "dataset_sup/"

# Model parameters
batch_size = 32

# Generators
train_datagen = Data_Generator(type="train",
                               batch_size=batch_size,
                               img_h=img_h,
                               img_w=img_w)

test_datagen = Data_Generator(type="test",
                              batch_size=batch_size,
                              img_h=img_h,
                              img_w=img_w)

# Convolution base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(34, 152, 1)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_datagen.generate(),
                              steps_per_epoch=10822,
                              epochs=1,
                              validation_data=test_datagen.generate(),
                              validation_steps=562)

model.save('tmp_model.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss') 
plt.plot(epochs, val_loss, 'b', label='Validation loss') 
plt.title('Training and validation loss')
plt.legend()
plt.show()