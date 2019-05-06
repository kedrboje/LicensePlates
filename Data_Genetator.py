import keras
import json
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


class Data_Generator():

    def __init__(self, batch_size, img_h, img_w, dirpath="dataset_sup/"):
        self.batch_size = batch_size
        self.img_w = img_w
        self.img_h = img_h
        self.train_dirpath = os.path.join(dirpath, "train/img/")
        self.test_dirpath = os.path.join(dirpath, "test/img")
        self.num_train = 10822
        self.num_test = 562
        self.current = 0
        self.voc = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "A", "B", "C", "E", "H", "K", "M", "O", "P", "T", "X", "Y"]
        self.imgs = np.zeros((self.num_test, self.img_h, self.img_w))
        self.labels = []

        for root, dirs, files in os.walk(self.test_dirpath):
            for i, file in enumerate(files):
                img = cv2.imread(os.path.join(self.test_dirpath, file), 0)
                img = img.astype(np.float32)
                img /= 255
                label = os.path.splitext(file)[0]
                self.imgs[i, :, :] = img
                self.labels.append(label)

        self.num = len(self.imgs)

    def _get_sample(self):
        self.current += 1
        if self.current >= self.num:
            self.current = 0
            random.randint(1, self.num)
        return self.imgs[self.current], self.labels[self.current]

    def encode(self, text):
        return list(map(lambda x: self.voc.index(x), text))

    def generate(self):
        while True:
            x = np.ones([self.batch_size, self.img_h, self.img_w, 1])
            y = np.ones([self.batch_size, 8])
            for i in range(self.batch_size):
                img, label = self._get_sample()
                img = np.expand_dims(img, -1)
                label = self.encode(label)
                x[i] = img
                y[i] = label
            yield (x, y)


datagen = Data_Generator(batch_size=32, img_h=34, img_w=152)

for i, (img, lbl) in enumerate(datagen.generate()):
    plt.imshow(img[i, :, :, 0], cmap='gray')
    plt.xlabel(lbl[i])
    plt.show()
    if i == 4:
        break
