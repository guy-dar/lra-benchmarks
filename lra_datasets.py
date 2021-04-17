import numpy as np
import pandas as pd
import pickle
from functools import reduce
import torch

class ListOpsDataset:
    def __init__(self, config):
        data_path = "datasets/lra_release/listops-1000/basic_train.tsv"
        self.data = pd.read_csv(data_path, delimiter='\t')
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
        
    def __getitem__(self, i):
        data = self.data.iloc[i]
        source = data.Source
        inputs = self.tokenizer(source, max_length=self.max_length) #return_tensors='pt', truncation=True, padding='max_length'
        target = data.Target
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data)

class Cifar10Dataset:
    def __init__(self, config):
        data_paths = [f"datasets/cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)]
        print("loading cifar-10 data...")
        data_dicts = [Cifar10Dataset.unpickle(path) for path in data_paths]
        print("assembling cifar-10 files..")
        self.data = reduce((lambda x, y: {b'data': np.concatenate([x[b'data'], y[b'data']], axis=0), 
                                         b'labels': np.concatenate([x[b'labels'], y[b'labels']], axis=0)}), 
                           data_dicts)
        # TODO CHECK: i think this is the right shape 
        # see: https://www.cs.toronto.edu/~kriz/cifar.html 
        #      section "Dataset layouts" discusses the memory layout of the array
        self.data[b'data'] = self.data[b'data'].reshape((-1, 3, 1024)) 
       
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
    
    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        return d
    
    def __getitem__(self, i):
        source = self.data[b'data'][i]
#         source = np.round(source.mean(axis=0)).astype(int)
        inputs = self.tokenizer(source, max_length=self.max_length)
        target = self.data[b'labels'][i]
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data[b'data'])
