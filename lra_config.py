import torch
import ml_collections

# helper fns

def make_char_tokenizer(allowed_chars, lowercase_input=False):
    # make distinct
    allowed_chars = list(set(allowed_chars))
    
    def _tokenizer(x, max_length):
        x = x[:max_length] # truncate
        if lowercase_input:
            x = x.lower()
        n = len(x)
        mask = ([1] * n) + ([0] * (max_length-n))
        ids = list(map(lambda c: allowed_chars.index(c)+1, x)) + ([0] * (max_length-n))
        return {'input_ids': torch.LongTensor([ids]), 
                'attention_mask': torch.LongTensor([mask])}
    
    _tokenizer.vocab_size = len(allowed_chars)+1
    return _tokenizer

def pixel_tokenizer(x, max_length):
    # note: x is not batched
    return x.view(-1)
pixel_tokenizer.vocab_size = 256 + 1
ascii_tokenizer = make_char_tokenizer(''.join(chr(i) for i in range(256)))

# configs

def get_listops_config():
    config = ml_collections.ConfigDict()
    config.batch_size = 4
#     config.eval_frequency = 20
#     config.num_eval_steps = 20
    config.total_train_samples = 160000
    config.learning_rate = 0.05
    config.weight_decay = 1e-1
    config.warmup = 1000
    config.max_length = 2000
    config.tokenizer = make_char_tokenizer(set('0123456789 MIN MAX MEDIAN SUM_MOD [ ] ( )'))

    model_config = ml_collections.ConfigDict()    
    model_config.max_position_embeddings = config.max_length
    model_config.num_attention_heads = 8
    model_config.num_hidden_layers = 6
    model_config.hidden_size = 512
    model_config.intermediate_size = 2048
    model_config.num_labels = 10
    model_config.vocab_size = config.tokenizer.vocab_size
    
    return config, model_config

def get_text_classification_config(num_labels=2):
    config = ml_collections.ConfigDict()
    config.batch_size = 4
#     config.eval_frequency = 100
    config.total_train_samples = 640000
    config.learning_rate = 0.05
    config.weight_decay = 1e-1
    config.warmup = 8000
    config.tokenizer = ascii_tokenizer
    config.max_length = 1000

    model_config = ml_collections.ConfigDict()
    model_config.max_position_embeddings = config.max_length
    model_config.num_attention_heads = 4
    model_config.num_hidden_layers = 4
    model_config.hidden_size = 256
    model_config.intermediate_size = 1024
    model_config.num_labels = num_labels
    model_config.vocab_size = config.tokenizer.vocab_size

    return config, model_config

def get_cifar10_config():
    NUM_EPOCHS = 200
    TRAIN_EXAMPLES = 45000
#     VALID_EXAMPLES = 10000
    
    config = ml_collections.ConfigDict()
    config.batch_size = 256
#     config.eval_frequency = TRAIN_EXAMPLES // config.batch_size
#     config.num_eval_steps = VALID_EXAMPLES // config.batch_size
    config.total_train_samples = TRAIN_EXAMPLES * NUM_EPOCHS
    config.weight_decay = 0.
    config.learning_rate = .0005
    config.warmup = (TRAIN_EXAMPLES // config.batch_size) * 1
#     config.steps_per_cycle = (TRAIN_EXAMPLES // config.batch_size) * NUM_EPOCHS

    # model params
    model_config = ml_collections.ConfigDict()
    model_config.max_position_embeddings = config.max_length
    model_config.hidden_size = 32
    config.model.num_attention_heads = 1
    config.model.num_hidden_layers = 1
    config.model.intermediate_dim = 64
    model_config.dropout_rate = 0.3
    model_config.attention_dropout_rate = 0.2
    model_config.num_labels = 10
    model_config.vocab_size = config.tokenizer.vocab_size
    
    return config, model_config
