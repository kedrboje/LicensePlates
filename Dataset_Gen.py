import keras
import json
import os
import numpy as np
import Alphabet_Gen as AG


class Dataset_gen():

    def __init__(self, tag: str, dirpath: str = 'dataset/'):

        self.num_sampes = 0
        self.alp = AG.Alphabet_Gen(tag)
        self.alp.get_alphabet()

        if tag == 'train':
            self.dirpath = os.path.join(dirpath, tag)
            self.ann_train_dirpath = os.path.join(dirpath, tag, 'ann')
            self.img_train_dirpath = os.path.join(dirpath, tag, 'img')

            for root, dirs, files in os.walk(self.ann_train_dirpath):
                for file in files:
                    ann = json.load(open(os.path.join(root, file), 'r'))
                    desc = ann['description']
                    if self.alp.is_valid(desc):
                        self.num_sampes += 1

        elif tag == 'test':
            self.dirpath = os.path.join(dirpath, tag)
            self.ann_test_dirpath = os.path.join(dirpath, tag, 'ann')
            self.img_test_dirpath = os.path.join(dirpath, tag, 'img')

            for root, dirs, files in os.walk(self.ann_test_dirpath):
                for file in files:
                    ann = json.load(open(os.path.join(root, file), 'r'))
                    desc = ann['description']
                    if self.alp.is_valid(desc):
                        self.num_sampes += 1


asd = Dataset_gen(tag='test')  # don't use train (see comment above)
qwe = Dataset_gen(tag='train')
print(asd.num_sampes + qwe.num_sampes)
print(qwe.num_sampes)
