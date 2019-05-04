import keras 
from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt
import json
import os 
from keras.preprocessing.image import ImageDataGenerator


#  Image Preprocessing

dataset_path = "dataset_sup/"

train_data = np.zeros()

#  Generators
# train_data_gen = ImageDataGenerator(rescale=1./255) 
# test_data_gen = ImageDataGenerator(rescale=1./255)

