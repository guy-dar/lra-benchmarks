from itertools import cycle
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import (AutoTokenizer, BertForSequenceClassification, BertConfig)
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.utils.data import DataLoader
from ml_collections import ConfigDict
from lra_config import (get_listops_config, get_cifar10_config)
import pickle
from functools import reduce

# helper fns
def dict_to_device(inputs, device):
    return {key: inputs[key].to(device) for key in inputs}

def transformers_collator(sample_list):
    input_list, target_list = zip(*sample_list)
    keys = input_list[0].keys()
    inputs = {k: torch.cat([inp[k] for inp in input_list], dim=0) for k in keys}
    target = torch.cat(target_list, dim=0) 
    return inputs, target

# datasets
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
        return len(self.data)

# tasks
TASKS = {
         'listops': ConfigDict(dict(dataset_fn=ListOpsDataset, config_getter=get_listops_config)),
         'cifar10': ConfigDict(dict(dataset_fn=Cifar10Dataset, config_getter=get_cifar10_config)),
        }
    
# fetch task 
task_name = "listops"
task = TASKS[task_name]

# config
config, model_config = task.config_getter()
model_config = BertConfig(**model_config)

# hyperparams
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = config.learning_rate
wd = config.weight_decay
batch_size = config.batch_size 
warmup_steps = config.warmup
max_train_steps = int(np.ceil(config.total_train_samples / config.batch_size)) # can keep it float

# model
model = BertForSequenceClassification(model_config)
model.to(device)
tokenizer = config.tokenizer
optimizer = Adam(model.parameters(), lr=lr, weight_decay = wd)
scheduler = MultiplicativeLR(optimizer, lambda step: step/warmup_steps if step < warmup_steps else step**(-.5))

# load data
dataset = task.dataset_fn(config)
dataloader = DataLoader(dataset, batch_size = batch_size, collate_fn=transformers_collator)

# train model
model.train()
running_loss = 0
running_acc = 0
pbar = tqdm(cycle(dataloader), total = max_train_steps)
for i, (inputs, target) in enumerate(pbar):
    if i == max_train_steps:
        break
    optimizer.zero_grad()
    inputs = dict_to_device(inputs, device)
    target = target.to(device)
    outputs = model(**inputs)
    loss = F.cross_entropy(outputs.logits, target)

    running_loss += loss.item()
    running_acc += sum(torch.argmax(loss, dim=-1) == target).item()/batch_size
    pbar.set_postfix_str(f"loss: {running_loss/(i+1):.2f} accuracy: {running_acc/(i+1):.2f}")

    loss.backward()
    optimizer.step()
    scheduler.step()
