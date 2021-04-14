import torch
from torch import nn
import transformers
from transformers import AutoModelForSequenceClassification, AutoModelForSequenceClassification
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.utils.data import DataLoader

class ListOpsDataset:
    def __init__(self, tokenizer):
        data_path = "datasets/listops-1000/basic_train.tsv"
        self.data = pd.read_csv(data_path, delimiter='\t')
        self.tokenizer = tokenizer
        
    def __getitem__(self, i):
        data = self.data.iloc[i]
        inputs = self.tokenizer(data.Source, return_tensors='pt')
        target = data.Target
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data)
    
# config
config = {
    "n_hidden_layers": 6,
    "max_length": 2048,    
}
# hyperparams
epochs = 1
lr = 1e-4
batch_size = 4

# model
model = AutoModelForSequenceClassification.from_config(**config)
tokenizer = AutoTokenizer.from_config(**config)
optimizer = Adam(model.parameters(), lr=lr)
scheduler = MultiplicativeLR(lambda x: 1)
# load data
datset = ListOpsDataset(tokenizer)
dataloader = DataLoader(dataset, batch_size = batch_size)

# train model
for _ in range(epochs):
    running_loss = 0
    pbar = tqdm(dataloader)
    for i, (inputs, target) in enumerate(pbar):
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs, target)
        loss.backward()
        running_loss += loss.item()
        pbar.set_postfix_str(f"loss: {running_loss/(i+1)}")
        optimizer.step()
        scheduler.step()