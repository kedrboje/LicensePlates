"""
Class Alphabet_Gen consist of two methods:
    1. get_alphabet() - return sorted array of symbols from the .json files in
    the dataset/test/ann or dataset/train/ann folders.
    2. is_valid(string) - return Bool value. True if string consist of
    characters from the generated alphabet.
"""
import keras
import os
import json
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

    def is_valid(self, string):

        for symbol in string:
            if symbol not in self.symbols:
                # raise ValueError
                return False
            else:
                return True

# gen = Alphabet_Gen(tag='train')
# print(gen.get_alphabet())
