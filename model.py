import keras
from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from keras.preprocessing.image import ImageDataGenerator
from Data_Genetator import Data_Generator
import keras.backend as K


# Image preprocessing
img_h = 64
img_w = 128
dirpath = "dataset_sup/"

# Model parameters
batch_size = 32

# Generators
train_datagen = Data_Generator(type="train",
                               batch_size=batch_size,
                               img_h=img_h,
                               img_w=img_w,
                               downsample=4,
                               max_lbl_len=9)

valid_datagen = Data_Generator(type="valid",
                               batch_size=batch_size,
                               img_h=img_h,
                               img_w=img_w,
                               downsample=4,
                               max_lbl_len=9)

conv_to_rnn = (img_w // 4, (img_h // 4) * 16)

input_sample = layers.Input(name='input', shape=(img_w, img_h, 1))

conv = layers.Conv2D(16, (3, 3), padding="same", activation="relu", kernel_initializer='he_normal')(input_sample)

max_pool_first = layers.MaxPooling2D((2, 2))(conv)

conv_2 = layers.Conv2D(16, (3, 3), padding="same", activation="relu", kernel_initializer='he_normal')(max_pool_first)

max_pool_2 = layers.MaxPooling2D((2, 2))(conv_2)

reshape = layers.Reshape(target_shape=conv_to_rnn)(max_pool_2)

first_dense = layers.Dense(32, activation='relu')(reshape)

gru = layers.GRU(512, return_sequences=True, kernel_initializer='he_normal')(first_dense)

gru_backwards = layers.GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(first_dense)

gru_merged = keras.layers.merge.add([gru, gru_backwards])

gru_2 = layers.GRU(512, return_sequences=True, kernel_initializer='he_normal')(gru_merged)

gru_2_backwards = layers.GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(gru_merged)

dense_2 = layers.Dense(23, kernel_initializer='he_normal')(keras.layers.merge.concatenate([gru_2, gru_2_backwards]))

act = layers.Activation('softmax', name='softmax')(dense_2)

labels = layers.Input(name='labels', shape=[train_datagen.max_lbl_len], dtype='float32')
input_len = layers.Input(name='input_len', shape=[1], dtype='int64')
label_len = layers.Input(name='label_len', shape=[1], dtype='int64')

def ctc_lambda_func(args):
    act, labels, input_len, label_len = args
    act = act[:, 2:, :]
    return K.ctc_batch_cost(labels, act, input_len, label_len)

loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([act, labels, input_len, label_len])

sgd = keras.optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

###

model = models.Model(inputs=[input_sample, labels, input_len, label_len], outputs=loss_out)

model.summary()


model.compile(optimizer=sgd,
              loss={'ctc': lambda y_true, act: act},
              metrics=['acc'])

test_func = K.function([input_sample], [act])

history = model.fit_generator(train_datagen.generate(),
                              steps_per_epoch=4392,
                              epochs=1,
                              validation_data=valid_datagen.generate(),
                              validation_steps=546)

model.save('sex_model.h5')

# serialize model to JSON
model_json = model.to_json()
with open("sex_model.json", "w") as f:
    f.write(model_json)

# serialize weights to HDF5
model.save_weights("sex_model_weights.h5")
print("Saved model to disk")


