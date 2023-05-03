import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import wandb
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random

# define the special tokens that stand for start of seq, end of seq, 
# an unknown symbol.
SOS_SYM = '@'
EOS_SYM = '$'
UNK_SYM = '!'
PAD_SYM = '%' # define a special token for padding - this helps with batch processing 

# function to load the 'cat' (= train/val/test) data of language 'lang'
def load_data(lang, cat):
    fcontents = open(f'aksharantar_sampled/{lang}/{lang}_{cat}.csv','r', encoding='utf-8').readlines()
    pairs = [tuple(l.strip().split(',')) for l in fcontents]
    x_data, y_data = list(map(list,zip(*pairs)))
    return x_data, y_data

# class for a language with useful functions.
class Language:
    def __init__(self, name):
        self.lname = name
    
    # function to create the vocabulary(set of tokens) using the words in 'data'
    # here, a token is either a special token or a lang character
    def create_vocabulary(self, *data):
        symbols = set()
        for wd in data:
            for c in wd:
                symbols.add(c)
        self.symbols = symbols
    
    # function to generate the index2sym (a number to a token) and 
    # sym2index (a token to a number) mappings using the vocabulary
    def generate_mappings(self):
        self.index2sym = {0: SOS_SYM, 1 : EOS_SYM, 2 : UNK_SYM, 3 : PAD_SYM}
        self.sym2index = {SOS_SYM : 0, EOS_SYM : 1, UNK_SYM : 2, PAD_SYM : 3}
        self.symbols = list(self.symbols)
        self.symbols.sort()

        for i, sym in enumerate(self.symbols):
            self.sym2index[sym] = i + 4
            self.index2sym[i+4] = sym
        
        self.num_tokens = len(self.index2sym.keys())
    
    # function to tokenize a word and convert all the tokens to
    # their corr. numbers using sym2index
    def convert_to_numbers(self, word):
        enc = [self.sym2index[SOS_SYM]]
        for ch in word:
            if ch in self.sym2index.keys():
                enc.append(self.sym2index[ch])
            else:
                enc.append(self.sym2index[UNK_SYM])
        enc.append(self.sym2index[EOS_SYM])
        return enc
    
    # convert a list of predictions (each prediction is a list of numbers)
    # to the corresponding list of words using index2sym
    # pred should be numpy array of shape (number_of_words, max_word_length)
    # tokens after EOS_SYM are discarded
    def convert_to_words(self, preds):
        num = preds.shape[0]
        words = [] 
        for i in range(num):
            wd = ''
            for idx in preds[i][1:]: # 1: -> ignore SOS token
                if torch.is_tensor(idx):
                    idx = idx.item()
                ch = self.index2sym[idx]
                if ch != EOS_SYM:
                    wd += ch
                else:
                    break
            words.append(wd)
        return words

    # get the number assigned to a token
    def get_index(self, sym):
        return self.sym2index[sym]
    
    # get the number of tokens in the vocabulary
    def get_size(self):
        return self.num_tokens
