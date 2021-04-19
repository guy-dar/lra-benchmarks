from itertools import cycle
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import (AutoTokenizer, BertForSequenceClassification, BertConfig, 
                         Trainer, TrainingArguments)

from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from ml_collections import ConfigDict
from lra_config import (get_listops_config, get_cifar10_config, get_text_classification_config)
from lra_datasets import (ListOpsDataset, Cifar10Dataset, ImdbDataset)
from argparse import ArgumentParser

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
    inputs.update({'labels': target})
    return inputs

# consts
OUTPUT_DIR = "output_dir/"
deepspeed_json = "ds_config.json"

TASKS = {
         'listops': ConfigDict(dict(dataset_fn=ListOpsDataset, config_getter=get_listops_config)),
         'cifar10': ConfigDict(dict(dataset_fn=Cifar10Dataset, config_getter=get_cifar10_config)),
         'imdb': ConfigDict(dict(dataset_fn=ImdbDataset, config_getter=get_text_classification_config)),
        }

# main functions
def get_model(config, model_config):
    model_config = BertConfig(**model_config)
    model = BertForSequenceClassification(model_config)

    if config.tied_weights:
        layer_base = model.bert.encoder.layer
        force_weight_sharing(layer_base)
    return model

def compute_metrics(eval_res):
    preds, labels = eval_res.preds, eval_res.labels
    return {"accuracy": (np.argmax(pred, axis=-1) == labels).mean()}
    
def train(model, config, use_deepspeed):
    num_devices = 1
    lr = config.learning_rate
    wd = config.weight_decay
    batch_size = config.batch_size 
    device_batch_size = int(np.ceil(batch_size/num_devices))
    warmup_steps = config.warmup
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    
    dataset = task.dataset_fn(config, split='train')
    eval_dataset = task.dataset_fn(config, split='eval')    
    max_train_steps = int(np.ceil(config.total_train_samples / batch_size))
    if config.total_eval_samples < 0:
        max_eval_steps = len(eval_dataset) // batch_size
    else:
        max_eval_steps = int(np.ceil(config.total_eval_samples / batch_size))
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    default_scheduler = LambdaLR(optimizer, lambda step: min(1, step/warmup_steps))
    scheduler = config.get('lr_scheduler', default_scheduler)
    
    # trainer
    trainer_args = TrainingArguments(output_dir=OUTPUT_DIR, overwrite_output_dir=True, do_train=True, do_eval=True, 
                                     per_device_train_batch_size=device_batch_size, per_device_eval_batch_size=device_batch_size,
                                     max_steps=max_train_steps, eval_steps=max_eval_steps, logging_steps=config.eval_frequency, 
                                     gradient_accumulation_steps=gradient_accumulation_steps,
                                     deepspeed=deepspeed_json if use_deepspeed else None)
    
    trainer = Trainer(model=model, args=trainer_args, data_collator=transformers_collator,
                      train_dataset=dataset, eval_dataset=eval_dataset, optimizers=(optimizer,scheduler), 
                      compute_metrics=compute_metrics)
    
    # train model
    trainer.train()
    
# main
if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--task", default="listops", choices=TASKS.keys(),
                       help="choose an LRA dataset from available options")
    parser.add_argument("--deepspeed", action="store_true",
                       help="use deepspeed optimization for better performance")
    args = parser.parse_args()
    task_name = args.task
    if args.deepspeed:
        import deepspeed
    
    task = TASKS[task_name]
    config, model_config = task.config_getter()    
    model = get_model(config, model_config)
    train(model, config, use_deepspeed=args.deepspeed)