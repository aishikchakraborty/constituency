import os
import ast
import json
import numpy as np
import argparse
import random
from collections import Counter
from collections import defaultdict
import _pickle as pickle
from tqdm import tqdm
import random
from transformers import BertTokenizer
import utils
import torch
from tqdm import tqdm
random.seed(0)
np.random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

parser = argparse.ArgumentParser(description='Preprocessing files')
parser.add_argument('--data', type=str, default='data/',help='location of the data corpus')
parser.add_argument('--output-b', type=str, default='do', help='type of output base transformation')
parser.add_argument('--output-nb', type=str, default='cleft', help='type of output non-base transformation')
parser.add_argument('--input-type', type=str, default='normal', help='token type headwords or normal words')
parser.add_argument('--max-sentlen', type=int, default=50,
                    help='maximum sentence length')
parser.add_argument('--held-out', action='store_true', help='is the target transformation held out')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
args = parser.parse_args()

with open(os.path.join(args.data, 'processed_train_' + args.input_type + '_' + args.output_b + '_' + str(args.held_out) + '.tsv'), 'r') as f:
    train_sentences = f.read().strip().split('\n')
with open(os.path.join(args.data, 'processed_test_' + args.input_type + '_' + args.output_nb + '_' + args.output_b + '.tsv'), 'r') as f:
    test_sentences = f.read().strip().split('\n')
with open(os.path.join(args.data, 'processed_dev_' + args.input_type + '_' + args.output_b + '_' + str(args.held_out) + '.tsv'), 'r') as f:
    val_sentences = f.read().strip().split('\n')

print(len(train_sentences))
print(len(test_sentences))
print(len(val_sentences))

def get_sentence_lists(sentences):
    x = []
    y = []
    const = []

    for i in range(len(sentences)):
        const.append(sentences[i][0])
        x.append(sentences[i][1])
        y.append(sentences[i][2])
    
    return x, const, y


if args.input_type == 'normal':
    random.shuffle(train_sentences)
    train_x, train_const, train_y = get_sentence_lists(train_sentences)
    test_x, test_const, test_y = get_sentence_lists(test_sentences)
    val_x, val_const, val_y = get_sentence_lists(val_sentences)

def save_binarized_splits():
    train = (train_x, train_y, train_const)
    test = (test_x, test_y, test_const)
    val = (val_x, val_y, val_const)

    pickle.dump(train, open(os.path.join(args.data, 'train_' + args.input_type + '_' + args.output_b + '_' + str(args.held_out) +'.pkl'), 'wb'))
    pickle.dump(test, open(os.path.join(args.data, 'test_'  + args.input_type + '_' + args.output_nb + '_' + args.output_b + '.pkl'), 'wb'))
    pickle.dump(val, open(os.path.join(args.data, 'val_' + args.input_type + '_' + args.output_b + '_' + str(args.held_out) + '.pkl'), 'wb'))


save_binarized_splits()
