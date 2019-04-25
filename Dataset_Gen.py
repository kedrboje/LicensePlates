import keras
import json
import os
import cv2
import numpy as np
import Alphabet_Gen as AG
import matplotlib.pyplot as plt


class Dataset_gen():

    def __init__(self, tag: str, img_h=34, img_w=152, dirpath: str = 'dataset/'):  # delete NONE

        self.patterns = []
        self.alp = AG.Alphabet_Gen(tag)
        self.alp.get_alphabet()
        self.img_h = img_h
        self.img_w = img_w

        if tag == 'train':
            self.dirpath = os.path.join(dirpath, tag)
            self.ann_train_dirpath = os.path.join(dirpath, tag, 'ann')
            self.img_train_dirpath = os.path.join(dirpath, tag, 'img')

            for root, dirs, files in os.walk(self.ann_train_dirpath):
                for file in files:
                    file_path = os.path.join(root, file)
                    ann = json.load(open(file_path, 'r'))
                    label = ann['description']
                    if self.alp.is_valid(label):
                        self.patterns.append([file_path, label])

        elif tag == 'test':
            self.dirpath = os.path.join(dirpath, tag)
            self.ann_test_dirpath = os.path.join(dirpath, tag, 'ann')
            self.img_test_dirpath = os.path.join(dirpath, tag, 'img')

            for root, dirs, files in os.walk(self.ann_test_dirpath):
                for file in files:
                    file_path = os.path.join(root, file)
                    ann = json.load(open(file_path, 'r'))
                    label = ann['description']
#################### АТТЕНШОН !!!!!!  КОСТЫЛЮМБА !!!!!! ####################
                    file_path = file_path.replace('ann', 'img')
#################### АТТЕНШОН !!!!!!  КОСТЫЛЮМБА !!!!!! ####################
                    file_path = file_path[:-5]
                    if self.alp.is_valid(label):
                        self.patterns.append([file_path, label])

    def compile(self):
        self.imgs = np.zeros((len(self.patterns), self.img_h, self.img_w))
        self.labels = []
        for i, (filepath, label) in enumerate(self.patterns):
            image = cv2.imread(filepath, 0)
            # if image:
            image = image.astype(np.float32)
            image /= 255
            self.imgs[i, :, :] = image
            self.labels.append(label)
            # else:
                # print(filepath)
            plt.imshow(image)
            plt.xlabel(label)
            plt.show()
        # print(self.imgs)


asd = Dataset_gen(tag='test')  # don't use train (see comment above)
# qwe = Dataset_gen(tag='train')
asd.compile()
# print(asd.patterns)
