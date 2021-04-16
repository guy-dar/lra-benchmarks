import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoConfig)
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.utils.data import DataLoader
from lra_config import get_listops_config

# helper fns
def dict_to_device(inputs, device):
    return {key: inputs[key].to(device) for key in inputs}

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
    
# config
config, model_config = get_listops_config()
model_name = "bert-base-uncased" # only thing that matters here is the architecture. everything else will be overriden 
model_config = AutoConfig.from_pretrained(model_name, **model_config)

# hyperparams
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 1
lr = config.learning_rate
wd = config.weight_decay
batch_size = 1 #TODO: allow batching 
warmup_steps = config.warmup

# model
model = AutoModelForSequenceClassification.from_config(model_config)
model.to(device)
tokenizer = config.tokenizer
optimizer = Adam(model.parameters(), lr=lr, weight_decay = wd)
scheduler = MultiplicativeLR(optimizer, lambda step: step/warmup_steps if step < warmup_steps else step**(-.5))

# load data
dataset = ListOpsDataset(config)
dataloader = dataset #DataLoader(dataset, batch_size = batch_size)

# train model
model.train()
for _ in range(epochs):
    running_loss = 0
    running_acc = 0
    pbar = tqdm(dataloader)
    for i, (inputs, target) in enumerate(pbar):
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