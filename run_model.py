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

# helper fns
def dict_to_device(inputs, device):
    return {key: inputs[key].to(device) for key in inputs}

# datasets
class ListOpsDataset:
    def __init__(self, tokenizer):
        data_path = "datasets/lra_release/listops-1000/basic_train.tsv"
        self.data = pd.read_csv(data_path, delimiter='\t')
        self.tokenizer = tokenizer
        
    def __getitem__(self, i):
        data = self.data.iloc[i]
        source = data.Source
        inputs = self.tokenizer(source, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
        target = data.Target
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data)
    
# config
model_name = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 10

# hyperparams
device = 'cuda'
epochs = 1
lr = 1e-4
batch_size = 1 #TODO: allow batching 

# model
model = AutoModelForSequenceClassification.from_config(config)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
optimizer = Adam(model.parameters(), lr=lr)
scheduler = MultiplicativeLR(optimizer, lambda x: 1)
# load data
dataset = ListOpsDataset(tokenizer)
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
        loss.backward()
        running_loss += loss.item()
        running_acc += sum(torch.argmax(loss, dim=-1) == target).item()/batch_size
        pbar.set_postfix_str(f"loss: {running_loss/(i+1):.2f} accuracy: {running_acc/(i+1):.2f}")
        optimizer.step()
        scheduler.step()