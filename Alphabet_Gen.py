import keras
import tensorflow as tf
import os
import json
import cv2
from collections import Counter


class Alphabet_Gen():

    def __init__(self, tag, dirpath: str = 'dataset/'):
        self.lplates = list()
        self.symbols = ''
        if tag == 'train' or tag == 'test':
            self.dirpath = os.path.join(dirpath, tag, 'ann')
            
    def get_alphabet(self):

        for root, dirs, files in os.walk(self.dirpath):
            for file in files:
                ann = json.load(open(os.path.join(root, file), 'r'))
                tag = ann['tags']
                desc = ann['description']
                self.symbols += desc
        return sorted(Counter(self.symbols).keys())


gen = Alphabet_Gen(tag='train')
print(gen.get_alphabet())