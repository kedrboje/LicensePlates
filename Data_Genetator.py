import keras
import json
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


class Data_Generator():

    def __init__(self, type, batch_size, img_h, img_w, downsample, dirpath="dataset_sup/"):
        self.batch_size = batch_size
        self.img_w = img_w
        self.img_h = img_h
        self.downsample = downsample
        if type is "train":
            self.dirpath = os.path.join(dirpath, "train/")
            self.num = 10821
        if type is "test":
            self.dirpath = os.path.join(dirpath, "test/")
            self.num = 561
        if type is "valid":
            self.dirpath = os.path.join(dirpath, "valid/")
            self.num = 21
        self.current = 0
        self.voc = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "A", "B", "C", "E", "H", "K", "M", "O", "P", "T", "X", "Y"]
        self.imgs = np.empty((self.num, self.img_h, self.img_w))
        self.labels = []
        self.indexes = list(range(self.num))

        for root, dirs, files in os.walk(self.dirpath):
            for i, file in enumerate(files):
                img = cv2.imread(os.path.join(self.dirpath, file), 0)
                img = cv2.resize(img, (self.img_w, self.img_h))
                img = img.astype(np.float32)
                img /= 255
                label = os.path.splitext(file)[0]
                self.imgs[i, :, :] = img
                self.labels.append(label)

    def _get_sample(self):
        self.current += 1
        if self.current >= self.num:
            self.current = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.current]], self.labels[self.indexes[self.current]]

    def encode(self, text):
        return list(map(lambda x: self.voc.index(x), text))

    def generate(self):
        while True:
            x = np.empty([self.batch_size, self.img_w, self.img_h, 1])
            y = np.empty([self.batch_size, 8])
            for i in range(self.batch_size):
                img, label = self._get_sample()
                img = img.T
                img = np.expand_dims(img, -1)
                label = self.encode(label)
                x[i] = img
                y[i] = label
            inputs = {
                'input': x,
                'labels': y,
                'input_len': np.ones((self.batch_size, 1)) * (self.img_w // self.downsample - 2),
                'label_len': np.empty((self.batch_size, 1))
            }
            outputs = {
                'ctc': np.empty([self.batch_size])
            }
            yield (inputs, outputs)

# datagen = Data_Generator(type="test", batch_size=32, img_h=34, img_w=152)
# print(len(datagen.imgs))
# print(len(datagen.labels))
# print(len(datagen.indexes))
# for i, (img, lbl) in enumerate(datagen.generate()):
#     plt.imshow(img[i, :, :, 0])
#     plt.xlabel(lbl[i])
#     plt.show()
#     if i == 4:
#         break
# for i, (a, b) in enumerate(datagen.generate()):
#     print(a[i], b[i])
#     i += 1
#     print(i)

