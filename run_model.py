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
from lra_datasets import (ListOpsDataset, Cifar10Dataset)

# helper fns

# TODO: bad
def force_weight_sharing(layers_list):
    for i in range(len(layers_list)):
        layers_list[i] = layers_list[0]

def dict_to_device(inputs, device):
    return {key: inputs[key].to(device) for key in inputs}

def transformers_collator(sample_list):
    input_list, target_list = zip(*sample_list)
    keys = input_list[0].keys()
    inputs = {k: torch.cat([inp[k] for inp in input_list], dim=0) for k in keys}
    target = torch.cat(target_list, dim=0) 
    return inputs, target

# tasks
TASKS = {
         'listops': ConfigDict(dict(dataset_fn=ListOpsDataset, config_getter=get_listops_config)),
         'cifar10': ConfigDict(dict(dataset_fn=Cifar10Dataset, config_getter=get_cifar10_config)),
        }

# main functions
def get_model(config, model_config):
    model_config = BertConfig(**model_config)
    model = BertForSequenceClassification(model_config)

    if config.tied_weights:
        layer_base = model.bert.encoder.layer
        force_weight_sharing(layer_base)
    return model

def train(model, config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = config.learning_rate
    wd = config.weight_decay
    batch_size = config.batch_size 
    warmup_steps = config.warmup
    
    dataset = task.dataset_fn(config, split='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=transformers_collator)
    eval_dataset = task.dataset_fn(config, split='eval')    
    max_train_steps = int(np.ceil(config.total_train_samples / batch_size))
    if config.total_eval_samples < 0:
        max_eval_steps = len(eval_dataset) // batch_size
    else:
        max_eval_steps = int(np.ceil(config.total_eval_samples / batch_size))
    
    tokenizer = config.tokenizer
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = MultiplicativeLR(optimizer, lambda step: step/warmup_steps if step < warmup_steps else step**(-.5))
    
    # train model
    model.to(device)
    model.train()
    running_loss = 0
    running_acc = 0
    pbar = tqdm(cycle(dataloader), total=max_train_steps)
    for i, (inputs, target) in enumerate(pbar):
        if i == max_train_steps:
            break
        optimizer.zero_grad()
        inputs = dict_to_device(inputs, device)
        target = target.to(device)
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_acc += sum(torch.argmax(outputs.logits, dim=-1) == target).item()/batch_size
        pbar.set_postfix_str(f"loss: {running_loss/(i+1):.2f} accuracy: {running_acc/(i+1):.2f}")
        
        # evaluate
        if (config.eval_frequency > 0) and  ((i+1) % config.eval_frequency == 0):
            model.eval()
            eval_running_loss = 0.
            eval_running_acc = 0.
            eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, 
                                         collate_fn=transformers_collator)
            eval_pbar = tqdm(eval_dataloader, total=max_eval_steps)
            for j, (inputs, target) in enumerate(eval_pbar):
                if j == max_eval_steps:
                    break
                inputs = dict_to_device(inputs, device)
                target = target.to(device)
                outputs = model(**inputs)
                loss = F.cross_entropy(outputs.logits, target)
                eval_running_loss += loss.item()
                eval_running_acc += sum(torch.argmax(outputs.logits, dim=-1) == target).item()/config.eval_batch_size
                pbar.set_postfix_str(f"loss: {eval_running_loss/(j+1):.2f} accuracy: {eval_running_acc/(j+1):.2f}")
            model.train()
        
# main
if __name__ == "__main__":
    task_name = "cifar10"
    task = TASKS[task_name]
    config, model_config = task.config_getter()    
    model = get_model(config, model_config)
    train(model, config)