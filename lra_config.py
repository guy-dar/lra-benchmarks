import torch
import ml_collections

# helper fns
def ascii_tokenizer(x, max_length):
    x = x[:max_length] # truncate
    n = len(x)
    mask = ([1] * n) + ([0] * (max_length-n))
    ids = list(map(lambda c: ord(c)+1, x)) + ([0] * (max_length-n))
    return {'input_ids': torch.LongTensor([ids]), 
            'attention_mask': torch.LongTensor([mask])}

ascii_tokenizer.vocab_size = 256 + 1 # i guess..

# configs
def get_listops_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
#     config.batch_size = 32
#     config.eval_frequency = 20
    config.num_train_steps = 5000
#     config.num_eval_steps = 20
    config.learning_rate = 0.05
    config.weight_decay = 1e-1
#     config.max_predict_token_length = 50
    config.warmup = 1000
    config.max_length = 2000
    config.tokenizer = ascii_tokenizer 
#     config.tied_weights = True

    model_config = ml_collections.ConfigDict()
    model_config.max_position_embeddings = config.max_length
    model_config.num_attention_heads = 8
    model_config.num_hidden_layers = 6
    model_config.hidden_size = 512
    model_config.intermediate_size = 2048
    model_config.num_labels = 10
    model_config.vocab_size = ascii_tokenizer.vocab_size
    return config, dict(model_config)