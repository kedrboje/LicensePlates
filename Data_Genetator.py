import keras
import json
import os
import cv2
import numpy as np
import Alphabet_Gen as AG
import matplotlib.pyplot as plt


class Data_Generator():

    def __init__(self, batch_size, img_h, img_w, dirpath="dataset_sup/"):
        self.batch_size = batch_size
        self.img_size = (img_h, img_w)
        self.train_dirpath = os.path.join(dirpath, "train/img/")
        self.test_dirpath = os.path.join(dirpath, "test/img")
        self.samples = []

        for root, dirs, files in os.walk(self.test_dirpath):
            for file in files:
                img = cv2.imread(os.path.join(self.test_dirpath, file), 0)
                img = img.astype(np.float32)
                img /= 255
                label = os.path.splitext(file)[0]
                self.samples.append((img, label))
        print(self.samples)

    def generate(self):
        pass
        yield (x, y)
    

datagen = Data_Generator(batch_size=32, img_h=256, img_w=256)