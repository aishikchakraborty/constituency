from ast import parse
import numpy as np
import random
import os
import csv
import json
import time
import argparse
import _pickle as pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import models
import logging
import wandb
from transformers import BertTokenizer
from argparse import Namespace

import os

os.environ["WANDB_API_KEY"] = "75896b1ec51fe81d94a454defeb033dd0180a941"
os.environ["WANDB_MODE"] = "offline"


# wandb login 75896b1ec51fe81d94a454defeb033dd0180a941

parser = argparse.ArgumentParser(description='Generation Project Main')
parser.add_argument('--data', type=str, default='data/', help='location of the data corpus')
parser.add_argument('--output-b', type=str, default='do', help='base transformations')
parser.add_argument('--output-nb', type=str, default='cleft', help='non-base transformations')
parser.add_argument('--input-type', type=str, default='normal', help='type of operation to do on the data')
parser.add_argument('--encoder-type', type=str, default='bert', help='type of encoder')
parser.add_argument('--decoder-type', type=str, default='vanilla', help='type of decoder')
parser.add_argument('--decoding-type', type=str, default='top-k|beam', help='type of decoder')
parser.add_argument('--encoding-type', type=str, default='bert', help='type of encoder')
parser.add_argument('--load-checkpoint', type=str, help='load checkpoint for finetuning')
parser.add_argument('--random_seed', type=int, default=13370, help='random seed')
parser.add_argument('--numpy_seed', type=int, default=1337, help='numpy random seed')
parser.add_argument('--torch_seed', type=int, default=133, help='pytorch random seed')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--grad-clip', type=float, default=0.25, help='Gradient clipping')
parser.add_argument('--batch-size', type=int, default=32, help='Batch Size')
parser.add_argument('--epochs', type=int, default=10, help='Epochs')
parser.add_argument('--enc-hidden-dim', type=int, default=768, help='Hidden Dimensions of the encoder')
parser.add_argument('--dec-hidden-dim', type=int, default=768, help='Hidden Dimensions of the decoder')
parser.add_argument('--poshidden-dim', type=int, default=30, help='Hidden Dimensions of the pos decoder')
parser.add_argument('--posembed-dim', type=int, default=20, help='Pos embedding size')
parser.add_argument('--alpha', type=float, default=0.2, help='Degree of distillation')
parser.add_argument('--num-layers', type=int, default=1, help='Number of layers of the sequence model')
parser.add_argument('--gpu', type=int, default=0, help='GPU id')
parser.add_argument('--maxlen', type=int, default=15, help='Maxlen for sentences')
parser.add_argument('--top-k', type=int, default=5, help='Top-k sampling')
parser.add_argument('--top-p', type=float, default=0.9, help='Nucleus sampling')
parser.add_argument('--temp', type=float, default=5, help='Softmax Temperature')
parser.add_argument('--embed-dim', type=int, default=768, help='Decoder embedding dimensions')
parser.add_argument('--log-interval', type=int, default=40, help='Print interval')
parser.add_argument('--accelorator', type=str, default ='gpu', help='use CUDA')
parser.add_argument('--start-temp', type=float, default=1.5, help='Starting softmax temperature')
parser.add_argument('--min-temp', type=float, default=0.5, help='Minimum softmax temperature')
parser.add_argument('--num-accumulations', type=int, default=10, help='Number of gradient accumulations')
parser.add_argument('--cuda', action='store_true', help='use cuda')
# parser.add_argument('--tie-weights', action='store_true', help='tie output weights')
# parser.add_argument('--copy', action='store_true', help='use copynet')
parser.add_argument('--held-out', action='store_true', help='target transformation is held out or not')
parser.add_argument('--save', type=str, default='outputs/', help='output directory')
parser.add_argument('--num-constructions', type=int, default=23, help='Number of output costructions')



# wandb_run_name = "run_" + str(args.hidden_dim) + '_' + str(args.poshidden_dim) + '_pos_' + str(args.use_pos) + '_enc-type_' + str(args.encoder_type) + '_dec-type_' + str(args.decoder_type) + '_mixed_' + str(args.mixed) + '_augmented_' + str(args.augmented)
wandb_logger = loggers.WandbLogger(project="const", log_model="all")
# tb_logger = TensorBoardLogger(save_dir="outputs/")
class GloveTokenizer():
    def __init__(self):
        self.pad_token_id = 1
        self.cls_token_id = 3
        self.sep_token_id = 2
        self.unk_token_id = 0

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, force_download=True)
vocab = tokenizer.get_vocab()
vectors = []
print(len(vocab))

parser.add_argument("--input-dim", type=int, default=len(vocab))
parser.add_argument("--output-dim", type=int, default=len(vocab))


early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='max'
)
parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()
print(args)


output_file_names = f'model_{args.encoder_type}_decoder_{args.decoder_type}_output_b_{args.output_b}_output_b_{args.output_b}_held_out_{args.held_out}_input_{args.input_type}'
checkpoint_callback = ModelCheckpoint(
    save_weights_only=True,
    monitor="val_loss",
    mode="min"
    )


trainer = pl.Trainer(
    default_root_dir=os.path.join(args.save, output_file_names),
    logger=wandb_logger,
    callbacks=[checkpoint_callback], 
    accelerator="gpu",
    devices=1 if torch.cuda.is_available() else None,
    progress_bar_refresh_rate=1,
    check_val_every_n_epoch=1,
    gradient_clip_val=args.grad_clip, 
    max_epochs=args.epochs,
    log_every_n_steps=1,
    num_sanity_val_steps=0
    )

trainer.logger._default_hp_metric = None 
# trainer.from_argparse_args(args)
dict_args = vars(args)


if args.encoder_type == 'bert':
    model = models.BERTRND(**dict_args)

pl.seed_everything(args.random_seed)

w2idx = {w:idx for idx, w in enumerate(vocab)}
idx2w = {idx:w for idx, w in enumerate(vocab)}

with open(os.path.join(args.data, 'processed_train_' + args.input_type + '_' + args.output_b + '_' + str(args.held_out) + '.tsv'), 'r') as f:
    train_sentences = f.read().strip().split('\n')
with open(os.path.join(args.data, 'processed_test_' + args.input_type + '_' + args.output_nb + '_' + args.output_b + '.tsv'), 'r') as f:
    test_sentences = f.read().strip().split('\n')
with open(os.path.join(args.data, 'processed_dev_' + args.input_type + '_' + args.output_b + '_' + str(args.held_out) + '.tsv'), 'r') as f:
    val_sentences = f.read().strip().split('\n')

def get_sentence_lists(sentences):
    x = []
    y = []
    const = []
    for i in range(len(sentences)):
        sentence_split = sentences[i].split(',')
    
        const.append(int(sentence_split[0]))
        x.append(sentence_split[1])
        y.append(sentence_split[2])
    
    return x, const, y


def detach_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def pad_sequences(s, pad_token):
    lengths = [len(s1) for s1 in s]
    print(f"Max length: {max(lengths)} Avg length: {sum(lengths)/len(s)}")

    longest_sent = args.maxlen
    

    padded_X = np.ones((len(s), longest_sent), dtype=np.int64) * tokenizer.pad_token_id
    # masked_X = np.ones((args.batch_size, longest_sent), dtype=np.int64) * pad_token
    for i, x_len in enumerate(lengths):
        sequence = s[i]
        padded_X[i, 0:x_len] = sequence[:x_len][:args.maxlen]
    
    masked_X = 1 - (padded_X == pad_token)

    return padded_X, masked_X



# criterion = nn.NLLLoss(reduction = 'sum')

def get_sentence(sentence):
    def _eos_parsing(sentence):
        if '[SEP]' in sentence:
            return sentence[:sentence.index('[SEP]')+1]
        else:
            return sentence
    # index sentence to word sentence
    sentence = [vocab[s] for s in sentence]

    return _eos_parsing(sentence)


def train_dataloader():
    train_x_processed = {}
    train_y_processed = {}
    train_x_processed_extended = {}
    train_y_processed_extended = {}

    if args.encoding_type == 'bert' :
        
        if args.input_type == 'lastword':
            tokenized_x = [s.split() for s in train_x]
        else:
            tokenized_x = [tokenizer.tokenize(s) for s in train_x]
        
        ids_x = [[tokenizer.cls_token_id] + [w2idx.get(s, tokenizer.unk_token_id) for s in sent] + [tokenizer.sep_token_id] for sent in tokenized_x]
        ids_extended_x = [[i + len(vocab) if s==tokenizer.unk_token_id else s for i, s in enumerate(sent)] for sent in ids_x]

        if args.input_type == 'lastword':
            tokenized_y = [s.split() for s in train_y]
        else:
            tokenized_y = [tokenizer.tokenize(s) for s in train_y]
        assert len(tokenized_x) == len(tokenized_y), "dimension mismatch"
        
        ids_y = [[tokenizer.cls_token_id] + [w2idx.get(s, tokenizer.unk_token_id) for s in sent] + [tokenizer.sep_token_id] for sent in tokenized_y]
        
        ids_extended_y = [[tokenizer.cls_token_id] + [w2idx.get(s, next((pos+len(vocab) for pos, inp_token in enumerate(tokenized_x[i]) if inp_token == s), tokenizer.unk_token_id)) for s in sent] + [tokenizer.sep_token_id] for i, sent in enumerate(tokenized_y)]    

        train_x_processed['input_ids'], train_x_processed['attention_mask'] = pad_sequences(ids_x, tokenizer.pad_token_id)
        train_y_processed['input_ids'], train_y_processed['attention_mask'] = pad_sequences(ids_y, tokenizer.pad_token_id)
        train_x_processed_extended['input_ids'], train_x_processed_extended['attention_mask'] = pad_sequences(ids_extended_x, tokenizer.pad_token_id)
        train_y_processed_extended['input_ids'], train_y_processed_extended['attention_mask'] = pad_sequences(ids_extended_y, tokenizer.pad_token_id)


    # breakpoint()
    train_dataset = TensorDataset(torch.tensor(train_x_processed['input_ids'], dtype=torch.long),
                        torch.tensor(train_x_processed['attention_mask'], dtype=torch.long),
                        torch.tensor(train_const, dtype=torch.long),
                        torch.tensor(train_y_processed['input_ids'], dtype=torch.long),
                        torch.tensor(train_x_processed_extended['input_ids'], dtype=torch.long),
                        torch.tensor(train_y_processed_extended['input_ids'], dtype=torch.long)
                        )
    # train loader
    train_sampler = RandomSampler(train_dataset)
    batched_train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=1)
    return batched_train_dataloader


def val_dataloader():
    val_x_processed = {}
    val_y_processed = {}
    val_x_processed_extended = {}
    val_y_processed_extended = {}

    if args.input_type == 'lastword':
        tokenized_x = [s.split() for s in val_x]
    else:
        tokenized_x = [tokenizer.tokenize(s) for s in val_x]
    ids_x = [[tokenizer.cls_token_id] + [w2idx.get(s, tokenizer.unk_token_id) for s in sent] + [tokenizer.sep_token_id] for sent in tokenized_x]
    ids_extended_x = [[i + len(vocab) if s==tokenizer.unk_token_id else s for i, s in enumerate(sent)] for sent in ids_x]

    if args.input_type == 'lastword':
        tokenized_y = [s.split() for s in val_y]
    else:
        tokenized_y = [tokenizer.tokenize(s) for s in val_y]
    ids_y = [[tokenizer.cls_token_id] + [w2idx.get(s, tokenizer.unk_token_id) for s in sent] + [tokenizer.sep_token_id] for sent in tokenized_y]
    ids_extended_y = [[tokenizer.cls_token_id] + [w2idx.get(s, next((pos+len(vocab)+1 for pos, inp_token in enumerate(tokenized_x[i]) if inp_token == s), tokenizer.unk_token_id)) for s in sent] + [tokenizer.sep_token_id] for i, sent in enumerate(tokenized_y)]    
    val_x_processed['input_ids'], val_x_processed['attention_mask'] = pad_sequences(ids_x, tokenizer.pad_token_id)
    val_y_processed['input_ids'], val_y_processed['attention_mask'] = pad_sequences(ids_y, tokenizer.pad_token_id)
    val_x_processed_extended['input_ids'], val_x_processed_extended['attention_mask'] = pad_sequences(ids_extended_x, tokenizer.pad_token_id)
    val_y_processed_extended['input_ids'], val_y_processed_extended['attention_mask'] = pad_sequences(ids_extended_y, tokenizer.pad_token_id)
    
    


    val_dataset = TensorDataset(torch.tensor(val_x_processed['input_ids'], dtype=torch.long), 
                        torch.tensor(val_x_processed['attention_mask'], dtype=torch.long),
                        torch.tensor(val_const, dtype=torch.long),
                        torch.tensor(val_y_processed['input_ids'], dtype=torch.long),
                        torch.tensor(val_x_processed_extended['input_ids'], dtype=torch.long),
                        torch.tensor(val_y_processed_extended['input_ids'], dtype=torch.long)
                        )
    # val loader
    # val_sampler = RandomSampler(val_dataset)
    batched_val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)
    return batched_val_dataloader


def test_dataloader():
    test_x_processed = {}
    test_y_processed = {}
    test_x_processed_extended = {}
    test_y_processed_extended = {}

    if args.encoding_type == 'bert' or args.encoding_type == 'roberta':
        # val_x_processed = tokenizer.batch_encode_plus(test_x, max_length=args.maxlen, pad_to_max_length=True)
        # val_y_processed = tokenizer.batch_encode_plus(test_y, max_length=args.maxlen, pad_to_max_length=True)
        if args.input_type == 'lastword':
            tokenized_x = [s.split() for s in test_x]
        else:
            tokenized_x = [tokenizer.tokenize(s) for s in test_x]
        ids_x = [[tokenizer.cls_token_id] + [w2idx.get(s, tokenizer.unk_token_id) for s in sent] + [tokenizer.sep_token_id] for sent in tokenized_x]
        ids_extended_x = [[i + len(vocab) if s==tokenizer.unk_token_id else s for i, s in enumerate(sent)] for sent in ids_x]

        if args.input_type == 'lastword':
            tokenized_y = [s.split() for s in test_y]
        else:
            tokenized_y = [tokenizer.tokenize(s) for s in test_y]
        ids_y = [[tokenizer.cls_token_id] + [w2idx.get(s, tokenizer.unk_token_id) for s in sent] + [tokenizer.sep_token_id] for sent in tokenized_y]
        ids_extended_y = [[tokenizer.cls_token_id] + [w2idx.get(s, next((pos+len(vocab)+1 for pos, inp_token in enumerate(tokenized_x[i]) if inp_token == s), tokenizer.unk_token_id)) for s in sent] + [tokenizer.sep_token_id] for i, sent in enumerate(tokenized_y)]    
        test_x_processed['input_ids'], test_x_processed['attention_mask'] = pad_sequences(ids_x, tokenizer.pad_token_id)
        test_y_processed['input_ids'], test_y_processed['attention_mask'] = pad_sequences(ids_y, tokenizer.pad_token_id)
        test_x_processed_extended['input_ids'], test_x_processed_extended['attention_mask'] = pad_sequences(ids_extended_x, tokenizer.pad_token_id)
        test_y_processed_extended['input_ids'], test_y_processed_extended['attention_mask'] = pad_sequences(ids_extended_y, tokenizer.pad_token_id)

    

    val_dataset = TensorDataset(torch.tensor(test_x_processed['input_ids'], dtype=torch.long),
                        torch.tensor(test_x_processed['attention_mask'], dtype=torch.long),
                        torch.tensor(test_const, dtype=torch.long),
                        torch.tensor(test_y_processed['input_ids'], dtype=torch.long), 
                        torch.tensor(test_x_processed_extended['input_ids'], dtype=torch.long),
                        torch.tensor(test_y_processed_extended['input_ids'], dtype=torch.long)
                        )

    # val loader
    # val_sampler = RandomSampler(val_dataset)
    batched_val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)
    return batched_val_dataloader

random.shuffle(train_sentences)
train_x, train_const, train_y = get_sentence_lists(train_sentences)
test_x, test_const, test_y = get_sentence_lists(test_sentences)
val_x, val_const, val_y = get_sentence_lists(val_sentences)

train_data = train_dataloader()
val_data = val_dataloader()
test_data = test_dataloader()


print(model.logger)

trainer.fit(model, train_data, val_data)
model = models.BERTRND.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
model.eval()
trainer.test(model, test_data)