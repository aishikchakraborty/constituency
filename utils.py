import io
import os
import json
import numpy as np
import argparse
import random
import re
from collections import Counter
from collections import defaultdict
import _pickle as pickle
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
import random
import torch
import ray
import dill
from transformers import BertTokenizer
from pathlib import Path
from joblib import Parallel, delayed
from stanza.server import CoreNLPClient
from unidecode import unidecode
from nltk.tree import Tree

random.seed(0)


def get_lastword_sentence(sentence, tokenizer):
    def _eos_parsing(sentence):
        if '[SEP]' in sentence:
            return sentence[:sentence.index('[SEP]')+1]
        else:
            return sentence
    # index sentence to word sentence
    _sentence = []
    
    _sentence = tokenizer.convert_ids_to_tokens(sentence)
    # sentence = [vocab[s] for s in sentence]
    return ' '.join(_eos_parsing(_sentence))

def get_sentence(sentence, vocab):
    def _eos_parsing(sentence):
        if '[SEP]' in sentence:
            return sentence[:sentence.index('[SEP]')+1]
        else:
            return sentence
    # index sentence to word sentence
    _sentence = []
    # print(len(src))
    # print(sentence)
    for s in sentence:
        if s >= len(vocab):
            _sentence.append(vocab[src[s - len(vocab)]])
        else:
            _sentence.append(vocab[s])
    # sentence = [vocab[s] for s in sentence]
    return ' '.join(_eos_parsing(_sentence))

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    # y_tensor = y.data
    y_tensor = y_tensor.long().contiguous().view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).to(y.device).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot
def remove_non_ascii(text):
    return unidecode(str(text))

def _trim_string(string):
    # remove extra spaces, remove trailing spaces, lower the case 
    return re.sub('\s+',' ',string).strip()

def dd():
    return 0

def get_glove_embeddings(unique_words):

    unk_cases = 0
    print('Reading Embeddings')
    f = open('data/glove/glove.42B.300d.txt', 'r')
    vocab = []
    embeddings = {}
    for lines in f:
        lines = lines.rstrip('\n').split()
        emb = [float(x) for x in lines[1:]]
        if lines[0] == 'unk':
            unk_vector = emb
        if len(emb) != 300:
            continue
        embeddings[lines[0]] = emb
    vocab = unique_words
    vectors = []
    for w in unique_words:
        try:
            vectors.append(embeddings[w])
        except:
            vectors.append(embeddings['unk'])
            unk_cases += 1
    
        
    vocab.insert(0, '[UNK]')
    vectors.insert(0, unk_vector)
    vocab.insert(1, '[PAD]')
    vectors.insert(1, [0.]*300)
    vocab.insert(2, '[SEP]')
    vectors.insert(2, [1.]*300)
    vocab.insert(3, '[CLS]')
    vectors.insert(3, [2.]*300)

    vectors = np.array(vectors)
    
    print('Read embeddings')
    print('Vocab Size: ', str(len(vocab)))
    print('unk cases: ', str(unk_cases))
    return vocab, vectors

def sample_dataset(text, parse, tense, total_samples=10000000000):
    # print(tense)
    present = [i for i in range(len(tense)) if tense[i]==0]
    past = [i for i in range(len(tense)) if tense[i]==1]
    random.shuffle(present)
    random.shuffle(past)
    parse_present = [parse[i] for i in present]
    parse_past = [parse[i] for i in past]
    present = [text[i] for i in present]
    past = [text[i] for i in past]

    total_samples = min(total_samples, min(len(present), len(past)))
    
    data = []
    idx_present = 0
    idx_past = 0
    print(len(present))
    print(len(past))
    while len(data) < total_samples:
        if random.random() < 0.5:
            data.append((present[idx_present], past[idx_past], parse_present[idx_present], parse_past[idx_past]))
            idx_present += 1
            idx_past += 1
        else:
            data.append((past[idx_past], present[idx_present], parse_past[idx_past], parse_present[idx_present]))
            idx_present += 1
            idx_past += 1
    return data


def get_head_words_and_constituents(dp, sentence):
    #  do_head : direct object head
    #  io_head : indirect object head
    tokens = sentence.split()
    
    # all possible connectors
    connectors = ['to', 'on', 'in', 'onto', 'into', 'with']
    connector_head = ""
    
    verb_heads = []
    is_cl = -1
    is_wh = -1
    is_flat = -1
    if tokens[0] == "what" or tokens[0] == "where" or tokens[0] == "who":
        is_wh = 1
    for dep in dp:
        if dep[1] == "it":
            is_cl = 1
        break
    for dep in dp: # sometimes verb heads are classified incorrectly
                   # some of these cases are addressed here
        if dep[2] == "acl:relcl" and is_cl == 1 :  
            verb_heads.append(dep[1])
        elif dep[2] == "flat":
            verb_head = dep[1]
            is_flat = 1
            break 
        elif dep[2] == "csubj" or dep[0] == "root":
            verb_heads.append(dep[1])
    
    if is_flat == -1:
        if len(verb_heads) > 1: # only in case of clefting
            do_head = verb_heads[0]
            verb_head = verb_heads[-1]
        else:
            verb_head = verb_heads[0]
        
    # print(sentence)
    subjs = []
    for dep in dp:
        if is_wh == 1:
            if dep[2] == "advmod":
                subj_head =  dep[1]
                subjs.append(dep[1])
        if dep[2] == "nsubj":
            subj_head = dep[1]
            subjs.append(dep[1])
    
    if len(subjs) > 1 and is_flat == 1:
        subj_head = subjs[0]
        do_head = subjs[1]
    

            
    
    object_heads = []
    for dep in dp:
        if dep[1] == "that" and dep[2] == "obl": # fixing inccorect parsing in case of clefting
            dep[2] = "obj"
        if is_cl == 1 and dep[2] == "root":
            object_heads.append(dep[1])
        if is_wh == 1 and dep[2] == "root":
            object_heads.append(dep[1])
        if is_wh == 1 and dep[2] == "advmod":
            object_heads.append(dep[1])
        if dep[2] == "obj" or dep[2] == 'obl:npmod' or dep[2] == "ccomp" or dep[2]=='obl:tmod':
            object_heads.append(dep[1])
    
    

    do_head = object_heads[0]
    

    attachment_idx = -1
    if len(object_heads) > 1:
        io_head = object_heads[1]
        for dep in dp:
            if dep[2] == "obl": # double object containing a prepositional object
                obl_idx = tokens.index(dep[1])
                # io_idx = tokens.index(io_head)
                # if obl_idx < io_idx :
                attachment_idx = obl_idx # this object attaches to the direct object
                break
    else:
        for dep in dp:
            if  dep[2] == "acl:relcl" and is_cl == -1:
                io_head = dep[1]    
                io_idx = tokens.index(io_head)
                do_idx = tokens.index(do_head)
                if do_idx > io_idx:
                    temp = do_head
                    do_head = io_head
                    io_head = temp
                break
            
            elif dep[2] == "obl" or dep[2] == 'iobj' or dep[2] == "obl:tmod" or dep[2] == "nmod:npmod" or dep[2] == "nmod" or dep[2] == "appos" : # first 'obl' is the indirect obj head
            
                io_head = dep[1]
                io_idx = tokens.index(io_head)
                do_idx = tokens.index(do_head)
                if do_idx > io_idx:
                    temp = do_head
                    do_head = io_head
                    io_head = temp
                break
            elif dep[2] == "root" and is_wh == 1:
                io_head = dep[1]
                io_idx = tokens.index(io_head)
                do_idx = tokens.index(do_head)
                if do_idx > io_idx:
                    temp = do_head
                    do_head = io_head
                    io_head = temp
                break
        
    for dep in dp:
        if dep[2] == "punct":
            punct = dep[1]
            break
    
    for dep in dp:
        if dep[1] in connectors: 
            connector_head = dep[1]
            break
    
    if connector_head and connector_head not in connectors:
        connector_head = ""

    head_words_map = {}
    head_words = []
    ignore_tokens_start = -1
    constituents = []
    for i, t in enumerate(tokens):
        if i == 0:
            current_phrase = tokens[i] + ' '
        if t == subj_head or t == verb_head:
            if i != 0:
                current_phrase += t + ' '
            constituents.append(current_phrase.strip())
            head_words.append(t)
            head_words_map[t] = current_phrase
            current_phrase = ''
        elif t == do_head:
            if i != 0:
                current_phrase += t + ' '
            if attachment_idx == -1:
                constituents.append(current_phrase.strip())
            
                head_words.append(t)
                head_words_map[t] = current_phrase
            else:
                current_phrase += ' '.join(tokens[(i+1):(attachment_idx+1)])
                ignore_tokens_start = i+1
                constituents.append(current_phrase.strip())
                head_words.append(t)
                head_words_map[t] = current_phrase
            current_phrase = ''
            
        elif t == io_head:
            current_phrase += ' '.join(tokens[i:-1])
            head_words.append(io_head)
            head_words_map[io_head] = current_phrase
            constituents.append(current_phrase.strip())
        
        elif t == connector_head:
            constituents.append(connector_head.strip())
            current_phrase = ''
            head_words.append(t)
            head_words_map[t] = t
            
        else:
            if ignore_tokens_start  == -1:
                current_phrase += t + ' '
            else:
                if i >= ignore_tokens_start and i < (attachment_idx+1):
                    continue 
                else:
                    current_phrase += t + ' '

    
    constituents.append(punct)
    head_words.append(punct)
    head_words_map[punct] = punct
    # if sentence == "michael passed the person across the table the salt .":
    #     breakpoint()
    
    return head_words, head_words_map, constituents



