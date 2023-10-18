import numpy as np
import torch
import random

def word_shuffle(x, y, p, tokenizer):
    x_ = []
    vocab = tokenizer.get_vocab()
    vocab_len = len(vocab)
    for i in range(x.size(0)):
        words = x[i].cpu().numpy().tolist()
        non_padded_words = [w for j, w in enumerate(words) if w!=tokenizer.pad_token_id and w!=tokenizer.cls_token_id and w!=tokenizer.sep_token_id]
        random.shuffle(non_padded_words)
        sent = [tokenizer.cls_token_id] + non_padded_words + [tokenizer.sep_token_id]
        sent += [tokenizer.pad_token_id] * (len(words)-len(sent))
        sent = sent[:len(words)]
        assert len(sent) == len(words), 'Weird Length ' + str(words) + ' ' + str(sent)
        # keep = np.random.rand(len(words)) > p
        # keep[0] = True
        # sent = [w for j, w in enumerate(words) if keep[j] and w!=tokenizer.pad_token_id]
        # sent += [tokenizer.pad_token_id] * (len(words)-len(sent))
        x_.append(sent)
    x_extended = [[i + len(vocab.keys()) if s==tokenizer.unk_token_id else s for i, s in enumerate(sent)] for sent in x_]
    y_extended = [[next((pos+len(vocab.keys()) for pos, inp_token in enumerate(x_[i]) if inp_token == s)) if s>=len(vocab.keys()) else s for i, s in enumerate(sent)] for sent in y]
    # breakpoint()
    x_ = torch.LongTensor(x_).to(x.device)
    x_extended = torch.LongTensor(x_extended).to(x.device)
    y_extended = torch.LongTensor(y_extended).to(y.device)
    return x_, x_extended, y_extended

def word_replace(x, y, p, tokenizer):
    x_ = []
    vocab = tokenizer.get_vocab()
    vocab_len = len(vocab)
    for i in range(x.size(0)):
        words = x[i].cpu().numpy().tolist()
        keep = np.random.rand(len(words)) > p
        keep[0] = True
        sent = [w if keep[j] and w!=tokenizer.pad_token_id else tokenizer.get_vocab()['[unused0]'] for j, w in enumerate(words)]
        sent += [tokenizer.pad_token_id] * (len(words)-len(sent))
        x_.append(sent)
    x_extended = [[i + len(vocab.keys()) if s==tokenizer.unk_token_id else s for i, s in enumerate(sent)] for sent in x_]
    y_extended = [[next((pos+len(vocab.keys()) for pos, inp_token in enumerate(x_[i]) if inp_token == s)) if s>=len(vocab.keys()) else s for i, s in enumerate(sent)] for sent in y]
    # breakpoint()
    x_ = torch.LongTensor(x_).to(x.device)
    x_extended = torch.LongTensor(x_extended).to(x.device)
    y_extended = torch.LongTensor(y_extended).to(y.device)
    return x_, x_extended, y_extended


def word_drop(x, y, p, tokenizer):
    x_ = []
    vocab = tokenizer.get_vocab()
    for i in range(x.size(0)):
        words = x[i].cpu().numpy().tolist()
        keep = np.random.rand(len(words)) > p
        keep[0] = False
        sent = [w for j, w in enumerate(words) if keep[j] and w!=tokenizer.pad_token_id]
        sent += [tokenizer.pad_token_id] * (len(words)-len(sent))
        x_.append(sent)
    x_extended = [[i + len(vocab.keys()) if s==tokenizer.unk_token_id else s for i, s in enumerate(sent)] for sent in x_]
    y_extended = [[next((pos+len(vocab.keys()) for pos, inp_token in enumerate(x_[i]) if inp_token == s)) if s>=len(vocab.keys()) else s for i, s in enumerate(sent)] for sent in y]
    # breakpoint()
    x_ = torch.LongTensor(x_).to(x.device)
    x_extended = torch.LongTensor(x_extended).to(x.device)
    y_extended = torch.LongTensor(y_extended).to(y.device)
    return x_, x_extended, y_extended

        
def noise(x, y, p, tokenizer, ntype='drop'):
    if ntype == 'drop':
        x, x_extended, y_extended = word_drop(x, y, p, tokenizer)
    if ntype == 'replace':
        x, x_extended, y_extended = word_replace(x, y, p, tokenizer)
    if ntype == 'shuffle':
        x, x_extended, y_extended = word_shuffle(x, y, p, tokenizer)
    return x, x_extended