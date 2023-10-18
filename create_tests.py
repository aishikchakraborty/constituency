import os
import json
import numpy as np
import argparse
import nltk
import random
from collections import Counter
from collections import defaultdict
import _pickle as pickle
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
import random
import pandas as pd
from transformers import DistilBertTokenizer
from pathlib import Path
from joblib import Parallel, delayed
import stanza
from stanza.server import CoreNLPClient
import utils
from transformers import BertTokenizer
from nltk.stem import WordNetLemmatizer 


lemmatizer = WordNetLemmatizer() 

random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Preprocessing nyu files')
parser.add_argument('--data', type=str, default='data/',
                    help='location of the data corpus')
parser.add_argument('--output-nb', type=str, default='c',
                    help='location of the data corpus')
parser.add_argument('--output-b', type=str, default='do',
                    help='location of the data corpus')
parser.add_argument('--held-out', action='store_true', help='target transformation is held-out')
parser.add_argument('--input-type', type=str, default='normal',
                    help='type of operation to do on the data')
parser.add_argument('--type', type=str, default='train',
                    help='location of the data corpus')
# parser.add_argument('--test', action='store_true', help='create test data')

args = parser.parse_args()
print(args)

class CreateConstituencySuite():
    def __init__(self, args):
        super().__init__()
        self._args = args
        self._client = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma, depparse', use_gpu=True, download_method=None)
        # self._client = CoreNLPClient(annotators=['tokenize','ssplit', 'pos', 'depparse'], timeout=10000000000, port=9000, memory='40G', threads=1, be_quiet=True, output_format='json', max_char_length=10000000000)
        self._alternations = ['dat', 'spray']
        self._alternations2idx = {a:idx for idx, a in enumerate(self._alternations)}
        self.transformations = ['do', 'po', 'loc', 'in', 's_cleft', 'do_cleft', 'io_cleft', 's_proform', 'do_proform', 'io_proform', 's_wh', 'do_wh', 'io_wh']
        self.transformations2idx = {t:idx for idx, t in enumerate(self.transformations)}

        if args.type == "train":
            self._artificial_subjects = self.create_artificial_subjects()
        else:
            self._artificial_subjects = None
        self._lemmatizer = WordNetLemmatizer() 

            
        # self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


    def create_artificial_subjects(self):
        names_corpus = nltk.corpus.names
        male_names = names_corpus.words('male.txt')
        male_names = [w.lower() for w in male_names]
        female_names = names_corpus.words('female.txt')
        female_names = [w.lower() for w in female_names]
        all_names = [w.lower() for w in male_names] + [w.lower() for w in female_names]
        random.shuffle(all_names)
        all_names = all_names[:30]
        return all_names


    def get_preprocessed_sentence(self, sentence):
        processed_sentence = self._client(sentence)
        # dependencies = [() for sent in processed_sentence]
        dependencies = [list((sent.words[word.head-1].text if word.head > 0 else "root", word.text, word.deprel)) for sent in processed_sentence.sentences for word in sent.words]
        head_words, head_words_map, constituents = utils.get_head_words_and_constituents(dependencies, sentence)
        return constituents, head_words, head_words_map
    

    def create_cleft(self, constituents):
        
        is_connector = 0
        for cons in constituents:
            if cons == 'to' or cons == 'on' or cons == 'in' or cons == 'onto' or cons == 'into' or cons == 'with':
                is_connector = 1
                break
        
        if is_connector:
            subj_cleft = constituents.copy()[:6]
            transformation = 'it was ' + constituents[0] + ' who'
            subj_cleft[0] = transformation
            
            s = constituents[0]
            v = constituents[1]
            do = constituents[2]
            connector = constituents[3]
            io = constituents[4:-1]
            subj_cleft[4] = ' '.join(io)


            do_cleft = constituents.copy()[:6]
            
            transformation = 'it was ' + do + ' that'
            do_cleft[0] = transformation
            do_cleft[1] = s
            do_cleft[2] = v
            do_cleft[3] = connector
            do_cleft[4] = ' '.join(io)


            
            io_cleft = constituents.copy()[:6]
            
            io_cleft[0] = 'it was ' + ' '.join(io) + ' that'
            io_cleft[1] = s
            io_cleft[2] = v
            io_cleft[3] = do
            io_cleft[4] = connector
        else:
            subj_cleft = constituents.copy()[:5]
            subj_cleft[0] = 'it was ' + constituents[0] + ' who'

            s = constituents[0]
            v = constituents[1]
            do = constituents[2]
            io = constituents[3:-1]
            subj_cleft[3] = ' '.join(io)


            io_cleft = constituents.copy()[:5]
            
            io_cleft[0] = 'it was ' + do + ' that'
            io_cleft[1] = s
            io_cleft[2] = v
            io_cleft[3] = ' '.join(io)

            do_cleft = constituents.copy()[:5]
            
            do_cleft[0] = 'it was ' + ' '.join(io) + ' that'
            do_cleft[1] = s
            do_cleft[2] = v
            do_cleft[3] = do

        return ' '.join(subj_cleft), ' '.join(do_cleft), ' '.join(io_cleft)

    def create_proform(self, constituents):
      
        is_connector = 0
        for cons in constituents:
            if cons == 'to' or cons == 'on' or cons == 'in' or cons == 'onto' or cons == 'into' or cons == 'with':
                is_connector = 1
                break
        
        if is_connector:
            subj_proform = constituents.copy()[:6]
            transformation = 'They'
            subj_proform[0] = transformation
            
            s = constituents[0]
            v = constituents[1]
            do = constituents[2]
            connector = constituents[3]
            io = constituents[4:-1]
            subj_proform[4] = ' '.join(io)


            do_proform = constituents.copy()[:6]
            
            transformation = 'that'
            do_proform[0] = s
            do_proform[1] = v
            do_proform[2] = transformation
            do_proform[3] = connector
            do_proform[4] = ' '.join(io)


            io_proform = constituents.copy()[:6]
            
            io_proform[0] = s
            io_proform[1] = v
            io_proform[2] = do
            io_proform[3] = connector
            io_proform[4] = 'them'
        else:
            subj_proform = constituents.copy()[:5]
            subj_proform[0] = 'they'

            s = constituents[0]
            v = constituents[1]
            do = constituents[2]
            io = constituents[3:-1]
            subj_proform[3] = ' '.join(io)


            io_proform = constituents.copy()[:5]
            
            io_proform[0] = s
            io_proform[1] = v
            io_proform[2] = 'them'
            io_proform[3] = ' '.join(io)

            do_proform = constituents.copy()[:5]
            
            do_proform[0] = s
            do_proform[1] = v
            do_proform[2] = do
            do_proform[3] = 'that'

        return ' '.join(subj_proform), ' '.join(do_proform), ' '.join(io_proform)
    
    def create_wh(self, constituents):
        
        is_connector_dat = 0
        is_connector_sl = 0

        for cons in constituents:
            if cons == 'to':
                is_connector_dat = 1
                break
        
        for cons in constituents:
            if cons == 'on' or cons == 'in' or cons == 'onto' or cons == 'into' or cons == 'with':
                is_connector_sl = 1
                connector_sl = cons
                break
        
        if is_connector_dat:
            subj_wh = constituents.copy()[:6]
            transformation = 'who'
            subj_wh[0] = transformation
            
            s = constituents[0]
            v = constituents[1]
            do = constituents[2]
            connector = constituents[3]
            io = constituents[4:-1]
            subj_wh[4] = ' '.join(io)


            do_wh = constituents.copy()[:6]
            
            transformation = 'what did'
            do_wh[0] = transformation
            do_wh[1] = s
            do_wh[2] = self._lemmatizer.lemmatize(v, 'v')
            do_wh[3] = connector
            do_wh[4] = ' '.join(io)


            
            io_wh = constituents.copy()[:6]
            
            io_wh[0] = 'who did'
            io_wh[1] = s
            io_wh[2] = self._lemmatizer.lemmatize(v, 'v')
            io_wh[3] = do
            io_wh[4] = connector
        
        elif is_connector_sl:
            subj_wh = constituents.copy()[:6]
            transformation = 'who'
            subj_wh[0] = transformation
            
            s = constituents[0]
            v = constituents[1]
            do = constituents[2]
            connector = constituents[3]
            io = constituents[4:-1]
            subj_wh[4] = ' '.join(io)

            do_wh = constituents.copy()[:6]
            
            if connector_sl == "with":
                
                transformation = 'what did'
                do_wh[0] = transformation
                do_wh[1] = s
                do_wh[2] = self._lemmatizer.lemmatize(v, 'v')
                do_wh[3] = do
                do_wh[4] = connector
                
            else:
                transformation = "what did"
                do_wh[0] = transformation
                do_wh[1] = s
                do_wh[2] = self._lemmatizer.lemmatize(v, 'v')
                do_wh[3] = connector
                do_wh[4] = ' '.join(io)


            
            io_wh = constituents.copy()[:6]
            if connector_sl == "with":
                transformation = 'where did'
                io_wh[0] = transformation
                io_wh[1] = s
                io_wh[2] = self._lemmatizer.lemmatize(v, 'v')
                io_wh[3] = ' '.join(io)
                io_wh[4] = ''
            else:
                transformation = "where did"
                io_wh[0] = transformation
                io_wh[1] = s
                io_wh[2] = self._lemmatizer.lemmatize(v, 'v')
                io_wh[3] = do
                io_wh[4] = connector
            
        else:

            subj_wh = constituents.copy()[:5]
            subj_wh[0] = 'who'

            s = constituents[0]
            v = constituents[1]
            do = constituents[2]
            io = constituents[3:-1]
            subj_wh[3] = ' '.join(io)


            io_wh = constituents.copy()[:5]
            
            io_wh[0] = 'who did'
            io_wh[1] = s
            io_wh[2] = self._lemmatizer.lemmatize(v, 'v')
            io_wh[3] = ' '.join(io)

            do_wh = constituents.copy()[:5]
            
            do_wh[0] = 'what did'
            do_wh[1] = s
            do_wh[2] = self._lemmatizer.lemmatize(v, 'v')
            do_wh[3] = do

        return ' '.join(subj_wh), ' '.join(do_wh), ' '.join(io_wh)

    def create_linguistic_tests(self):
        model_inputs = []
        sentence_idx = 0
        if self._args.type == 'train':
            f = open(os.path.join(self._args.data, self._args.type + '.tsv'), 'r')
        elif self._args.type == 'test' or self._args.type == 'dev':
            if self._args.output_b == 'do' or self._args.output_b == 'po':
                f = open(os.path.join(self._args.data, self._args.type + '_dative.tsv'), 'r')
            if self._args.output_b == 'in' or self._args.output_b == 'loc':
                f = open(os.path.join(self._args.data, self._args.type + '_sprayload.tsv'), 'r')
            
            
        
        sentences = []
        
        while True:
            line1 = f.readline()
            line2 = f.readline()
            
            sentence_idx += 2
            if sentence_idx %100 == 0:
                print("Preprocessed " + str(sentence_idx) + " sentences")

            if not line1 or not line2:
                break
            

            alt1 = line1.split('\t')    
            alt2 = line2.split('\t')

            if int(alt1[1]) == 0 or int(alt2[1]) == 0: # grammatically incorrect sentences
                continue
            if self._args.output_b == "do" or self._args.output_b == "po":
                if alt1[0] == "spray":
                    continue
            
            if self._args.output_b == "loc" or self._args.output_b == "in":
                if alt1[0] == "dat":
                    continue

            alteration_type = alt1[0]
            base_sentence1 = alt1[3].strip()
            base_sentence2 = alt2[3].strip()

        
            preprocessed_base_sentence1, head_words_base1, head_words_map_base1 =  self.get_preprocessed_sentence(base_sentence1)
            preprocessed_base_sentence2, head_words_base2, head_words_map_base2 =  self.get_preprocessed_sentence(base_sentence2)

        

            subj_cleft1, do_cleft1, io_cleft1 = self.create_cleft(preprocessed_base_sentence1)
            subj_cleft2, do_cleft2, io_cleft2 = self.create_cleft(preprocessed_base_sentence2)
            
            preprocessed_subj_cleft1, head_words_subj_cleft1, head_words_map_subj_cleft1 = self.get_preprocessed_sentence(subj_cleft1)
            preprocessed_do_cleft1, head_words_do_cleft1, head_words_map_do_cleft1 = self.get_preprocessed_sentence(do_cleft1)
            preprocessed_io_cleft1, head_words_io_cleft1, head_words_map_io_cleft1 = self.get_preprocessed_sentence(io_cleft1)

            preprocessed_subj_cleft2, head_words_subj_cleft2, head_words_map_subj_cleft2 = self.get_preprocessed_sentence(subj_cleft2)
            preprocessed_do_cleft2, head_words_do_cleft2, head_words_map_do_cleft2 = self.get_preprocessed_sentence(do_cleft2)
            preprocessed_io_cleft2, head_words_io_cleft2, head_words_map_io_cleft2 = self.get_preprocessed_sentence(io_cleft2)

            subj_proform1, do_proform1, io_proform1 = self.create_proform(preprocessed_base_sentence1)
            subj_proform2, do_proform2, io_proform2 = self.create_proform(preprocessed_base_sentence2)
            
            preprocessed_subj_proform1, head_words_subj_proform1, head_words_map_subj_proform1 = self.get_preprocessed_sentence(subj_proform1)
            preprocessed_do_proform1, head_words_do_proform1, head_words_map_do_proform1 = self.get_preprocessed_sentence(do_proform1)
            preprocessed_io_proform1, head_words_io_proform1, head_words_map_io_proform1 = self.get_preprocessed_sentence(io_proform1)

            preprocessed_subj_proform2, head_words_subj_proform2, head_words_map_subj_proform2 = self.get_preprocessed_sentence(subj_proform2)
            preprocessed_do_proform2, head_words_do_proform2, head_words_map_do_proform2 = self.get_preprocessed_sentence(do_proform2)
            preprocessed_io_proform2, head_words_io_proform2, head_words_map_io_proform2 = self.get_preprocessed_sentence(io_proform2)


            subj_wh1, do_wh1, io_wh1 = self.create_wh(preprocessed_base_sentence1)
            subj_wh2, do_wh2, io_wh2 = self.create_wh(preprocessed_base_sentence2)
            
            preprocessed_subj_wh1, head_words_subj_wh1, head_words_map_subj_wh1 = self.get_preprocessed_sentence(subj_wh1)
            preprocessed_do_wh1, head_words_do_wh1, head_words_map_do_wh1 = self.get_preprocessed_sentence(do_wh1)
            preprocessed_io_wh1, head_words_io_wh1, head_words_map_io_wh1 = self.get_preprocessed_sentence(io_wh1)

            preprocessed_subj_wh2, head_words_subj_wh2, head_words_map_subj_wh2 = self.get_preprocessed_sentence(subj_wh2)
            preprocessed_do_wh2, head_words_do_wh2, head_words_map_do_wh2 = self.get_preprocessed_sentence(do_wh2)
            preprocessed_io_wh2, head_words_io_wh2, head_words_map_io_wh2 = self.get_preprocessed_sentence(io_wh2)

            preprocessed_base_sentence1 = ' '.join(preprocessed_base_sentence1)
            preprocessed_base_sentence2 = ' '.join(preprocessed_base_sentence2)

            
            preprocessed_subj_cleft1 = ' '.join(preprocessed_subj_cleft1)
            preprocessed_do_cleft1 = ' '.join(preprocessed_do_cleft1)
            preprocessed_io_cleft1 = ' '.join(preprocessed_io_cleft1)
            
            preprocessed_subj_cleft2 = ' '.join(preprocessed_subj_cleft2)
            preprocessed_do_cleft2 = ' '.join(preprocessed_do_cleft2)
            preprocessed_io_cleft2 = ' '.join(preprocessed_io_cleft2)

            preprocessed_subj_proform1 = ' '.join(preprocessed_subj_proform1)
            preprocessed_do_proform1 = ' '.join(preprocessed_do_proform1)
            preprocessed_io_proform1 = ' '.join(preprocessed_io_proform1)

            preprocessed_subj_proform2 = ' '.join(preprocessed_subj_proform2)
            preprocessed_do_proform2 = ' '.join(preprocessed_do_proform2)
            preprocessed_io_proform2 = ' '.join(preprocessed_io_proform2)
            
            preprocessed_subj_wh1 = ' '.join(preprocessed_subj_wh1)
            preprocessed_do_wh1 = ' '.join(preprocessed_do_wh1)
            preprocessed_io_wh1 = ' '.join(preprocessed_io_wh1)

            preprocessed_subj_wh2 = ' '.join(preprocessed_subj_wh2)
            preprocessed_do_wh2 = ' '.join(preprocessed_do_wh2)
            preprocessed_io_wh2 = ' '.join(preprocessed_io_wh2)


            head_words_base1 = ' '.join(head_words_base1)
            head_words_base2 = ' '.join(head_words_base2)
            
            head_words_subj_cleft1 = ' '.join(head_words_subj_cleft1)
            head_words_do_cleft1 = ' '.join(head_words_do_cleft1)
            head_words_io_cleft1 = ' '.join(head_words_io_cleft1)

            head_words_subj_cleft2 = ' '.join(head_words_subj_cleft2)
            head_words_do_cleft2 = ' '.join(head_words_do_cleft2)
            head_words_io_cleft2 = ' '.join(head_words_io_cleft2)

            head_words_subj_proform1 = ' '.join(head_words_subj_proform1)
            head_words_do_proform1 = ' '.join(head_words_do_proform1)
            head_words_io_proform1 = ' '.join(head_words_io_proform1)

            head_words_subj_proform2 = ' '.join(head_words_subj_proform2)
            head_words_do_proform2 = ' '.join(head_words_do_proform2)
            head_words_io_proform2 = ' '.join(head_words_io_proform2)


            head_words_subj_wh1 = ' '.join(head_words_subj_wh1)
            head_words_do_wh1 = ' '.join(head_words_do_wh1)
            head_words_io_wh1 = ' '.join(head_words_io_wh1)

            head_words_subj_wh2 = ' '.join(head_words_subj_wh2)
            head_words_do_wh2 = ' '.join(head_words_do_wh2)
            head_words_io_wh2 = ' '.join(head_words_io_wh2)            

            # non-base transformations applied to base

            if self._args.output_b == "do":
                if self._args.type == 'test':
                    if self._args.output_nb == 'cleft':
                        model_inputs.append((self.transformations2idx['do'], subj_cleft1, subj_cleft2, head_words_subj_cleft1, head_words_subj_cleft2, head_words_map_subj_cleft1, head_words_map_subj_cleft2))
                        model_inputs.append((self.transformations2idx['do'], do_cleft1, do_cleft2, head_words_do_cleft1, head_words_do_cleft2, head_words_map_do_cleft1, head_words_map_do_cleft2))
                        model_inputs.append((self.transformations2idx['do'], io_cleft1, io_cleft2, head_words_io_cleft1, head_words_io_cleft2, head_words_map_io_cleft1, head_words_map_io_cleft2))
                    elif self._args.output_nb == 'proform':
                        model_inputs.append((self.transformations2idx['do'], subj_proform1, subj_proform2, head_words_subj_proform1, head_words_subj_proform2, head_words_map_subj_proform1, head_words_map_subj_proform2))
                        model_inputs.append((self.transformations2idx['do'], do_proform1, do_proform2, head_words_do_proform1, head_words_do_proform2, head_words_map_do_proform1, head_words_map_do_proform2))
                        model_inputs.append((self.transformations2idx['do'], io_proform1, io_proform2, head_words_io_proform1, head_words_io_proform2, head_words_map_io_proform1, head_words_map_io_proform2))
                    elif self._args.output_nb == 'wh':
                        model_inputs.append((self.transformations2idx['do'], subj_wh1, subj_wh2, head_words_subj_cleft1, head_words_subj_wh2, head_words_map_subj_wh1, head_words_map_subj_wh2))
                        model_inputs.append((self.transformations2idx['do'], do_wh1, do_wh2, head_words_do_wh1, head_words_do_wh2, head_words_map_do_wh1, head_words_map_do_wh2))
                        model_inputs.append((self.transformations2idx['do'], io_wh1, io_wh2, head_words_io_wh1, head_words_io_wh2, head_words_map_io_wh1, head_words_map_io_wh2))
                elif self._args.type == 'dev':
                    model_inputs.append((self.transformations2idx['do'], subj_cleft1, subj_cleft2, head_words_subj_cleft1, head_words_subj_cleft2, head_words_map_subj_cleft1, head_words_map_subj_cleft2))
                    model_inputs.append((self.transformations2idx['do'], do_cleft1, do_cleft2, head_words_do_cleft1, head_words_do_cleft2, head_words_map_do_cleft1, head_words_map_do_cleft2))
                    model_inputs.append((self.transformations2idx['do'], io_cleft1, io_cleft2, head_words_io_cleft1, head_words_io_cleft2, head_words_map_io_cleft1, head_words_map_io_cleft2))
                    
                    model_inputs.append((self.transformations2idx['do'], subj_proform1, subj_proform2, head_words_subj_proform1, head_words_subj_proform2, head_words_map_subj_proform1, head_words_map_subj_proform2))
                    model_inputs.append((self.transformations2idx['do'], do_proform1, do_proform2, head_words_do_proform1, head_words_do_proform2, head_words_map_do_proform1, head_words_map_do_proform2))
                    model_inputs.append((self.transformations2idx['do'], io_proform1, io_proform2, head_words_io_proform1, head_words_io_proform2, head_words_map_io_proform1, head_words_map_io_proform2))
                    
                    model_inputs.append((self.transformations2idx['do'], subj_wh1, subj_wh2, head_words_subj_cleft1, head_words_subj_wh2, head_words_map_subj_wh1, head_words_map_subj_wh2))
                    model_inputs.append((self.transformations2idx['do'], do_wh1, do_wh2, head_words_do_wh1, head_words_do_wh2, head_words_map_do_wh1, head_words_map_do_wh2))
                    model_inputs.append((self.transformations2idx['do'], io_wh1, io_wh2, head_words_io_wh1, head_words_io_wh2, head_words_map_io_wh1, head_words_map_io_wh2))

                else:
                    

                    model_inputs.append((self.transformations2idx['s_cleft'], preprocessed_base_sentence1, subj_cleft1, head_words_base1, head_words_subj_cleft1, head_words_map_base1, head_words_map_subj_cleft1))
                    model_inputs.append((self.transformations2idx['do_cleft'], preprocessed_base_sentence1, do_cleft1, head_words_base1, head_words_do_cleft1, head_words_map_base1, head_words_map_do_cleft1))
                    model_inputs.append((self.transformations2idx['io_cleft'], preprocessed_base_sentence1, io_cleft1, head_words_base1, head_words_io_cleft1, head_words_map_base1, head_words_map_io_cleft1))

                    model_inputs.append((self.transformations2idx['s_proform'], preprocessed_base_sentence1, subj_proform1, head_words_base1, head_words_subj_proform1, head_words_map_base1, head_words_map_subj_proform1))
                    model_inputs.append((self.transformations2idx['do_proform'], preprocessed_base_sentence1, do_proform1, head_words_base1, head_words_do_proform1, head_words_map_base1, head_words_map_do_proform1))
                    model_inputs.append((self.transformations2idx['io_proform'], preprocessed_base_sentence1, io_proform1, head_words_base1, head_words_io_proform1, head_words_map_base1, head_words_map_io_proform1))

                    model_inputs.append((self.transformations2idx['s_wh'], preprocessed_base_sentence1, subj_wh1, head_words_base1, head_words_subj_wh1, head_words_map_base1, head_words_map_subj_wh1))
                    model_inputs.append((self.transformations2idx['do_wh'], preprocessed_base_sentence1, do_wh1, head_words_base1, head_words_do_wh1, head_words_map_base1, head_words_map_do_wh1))
                    model_inputs.append((self.transformations2idx['io_wh'], preprocessed_base_sentence1, io_wh1, head_words_base1, head_words_io_wh1, head_words_map_base1, head_words_map_io_wh1))

                    model_inputs.append((self.transformations2idx['do'], preprocessed_base_sentence1, preprocessed_base_sentence2, head_words_base1, head_words_base2, head_words_map_base1, head_words_map_base2))
                    
                    model_inputs.append((self.transformations2idx['po'], subj_cleft2, subj_cleft1, head_words_subj_cleft2, head_words_subj_cleft1, head_words_map_subj_cleft2, head_words_map_subj_cleft1))
                    model_inputs.append((self.transformations2idx['po'], do_cleft2, do_cleft1, head_words_do_cleft2, head_words_do_cleft1, head_words_map_do_cleft2, head_words_map_do_cleft1))
                    model_inputs.append((self.transformations2idx['po'], io_cleft2, io_cleft1, head_words_io_cleft2, head_words_io_cleft1, head_words_map_io_cleft2, head_words_map_io_cleft1))

                    model_inputs.append((self.transformations2idx['po'], subj_proform2, subj_proform1, head_words_subj_proform2, head_words_subj_proform1, head_words_map_subj_proform2, head_words_map_subj_proform1))
                    model_inputs.append((self.transformations2idx['po'], do_proform2, do_proform1, head_words_do_proform2, head_words_do_proform1, head_words_map_do_proform2, head_words_map_do_proform1))
                    model_inputs.append((self.transformations2idx['po'], io_proform2, io_proform1, head_words_io_proform2, head_words_io_proform1, head_words_map_io_proform2, head_words_map_io_proform1))

                    model_inputs.append((self.transformations2idx['po'], subj_wh2, subj_wh1, head_words_subj_wh2, head_words_subj_wh1, head_words_map_subj_wh2, head_words_map_subj_wh1))
                    model_inputs.append((self.transformations2idx['po'], do_wh2, do_wh1, head_words_do_wh2, head_words_do_wh1, head_words_map_do_wh2, head_words_map_do_wh1))
                    model_inputs.append((self.transformations2idx['po'], io_wh2, io_wh1, head_words_io_wh2, head_words_io_wh1, head_words_map_io_wh2, head_words_map_io_wh1))

                    if not args.held_out:
                        model_inputs.append((self.transformations2idx['do'], subj_cleft1, subj_cleft2, head_words_subj_cleft1, head_words_subj_cleft2, head_words_map_subj_cleft1, head_words_map_subj_cleft2))
                        model_inputs.append((self.transformations2idx['do'], do_cleft1, do_cleft2, head_words_do_cleft1, head_words_do_cleft2, head_words_map_do_cleft1, head_words_map_do_cleft2))
                        model_inputs.append((self.transformations2idx['do'], io_cleft1, io_cleft2, head_words_io_cleft1, head_words_io_cleft2, head_words_map_io_cleft1, head_words_map_io_cleft2))
                        
                        model_inputs.append((self.transformations2idx['do'], subj_proform1, subj_proform2, head_words_subj_proform1, head_words_subj_proform2, head_words_map_subj_proform1, head_words_map_subj_proform2))
                        model_inputs.append((self.transformations2idx['do'], do_proform1, do_proform2, head_words_do_proform1, head_words_do_proform2, head_words_map_do_proform1, head_words_map_do_proform2))
                        model_inputs.append((self.transformations2idx['do'], io_proform1, io_proform2, head_words_io_proform1, head_words_io_proform2, head_words_map_io_proform1, head_words_map_io_proform2))
                        
                        model_inputs.append((self.transformations2idx['do'], subj_wh1, subj_wh2, head_words_subj_cleft1, head_words_subj_wh2, head_words_map_subj_wh1, head_words_map_subj_wh2))
                        model_inputs.append((self.transformations2idx['do'], do_wh1, do_wh2, head_words_do_wh1, head_words_do_wh2, head_words_map_do_wh1, head_words_map_do_wh2))
                        model_inputs.append((self.transformations2idx['do'], io_wh1, io_wh2, head_words_io_wh1, head_words_io_wh2, head_words_map_io_wh1, head_words_map_io_wh2))


            elif self._args.output_b == "po":
                if self._args.type == 'test':
                    if self._args.output_nb == 'cleft':
                        model_inputs.append((self.transformations2idx['po'], subj_cleft2, subj_cleft1, head_words_subj_cleft2, head_words_subj_cleft1, head_words_map_subj_cleft2, head_words_map_subj_cleft1))
                        model_inputs.append((self.transformations2idx['po'], do_cleft2, do_cleft1, head_words_do_cleft2, head_words_do_cleft1, head_words_map_do_cleft2, head_words_map_do_cleft1))
                        model_inputs.append((self.transformations2idx['po'], io_cleft2, io_cleft1, head_words_io_cleft2, head_words_io_cleft1, head_words_map_io_cleft2, head_words_map_io_cleft1))
                    elif self._args.output_nb == 'proform':
                        model_inputs.append((self.transformations2idx['po'], subj_proform2, subj_proform1, head_words_subj_proform2, head_words_subj_proform1, head_words_map_subj_proform2, head_words_map_subj_proform1))
                        model_inputs.append((self.transformations2idx['po'], do_proform2, do_proform1, head_words_do_proform2, head_words_do_proform1, head_words_map_do_proform2, head_words_map_do_proform1))
                        model_inputs.append((self.transformations2idx['po'], io_proform2, io_proform1, head_words_io_proform2, head_words_io_proform1, head_words_map_io_proform2, head_words_map_io_proform1))
                    elif self._args.output_nb == 'wh':
                        model_inputs.append((self.transformations2idx['po'], subj_wh2, subj_wh1, head_words_subj_wh2, head_words_subj_wh1, head_words_map_subj_wh2, head_words_map_subj_wh1))
                        model_inputs.append((self.transformations2idx['po'], do_wh2, do_wh1, head_words_do_wh2, head_words_do_wh1, head_words_map_do_wh2, head_words_map_do_wh1))
                        model_inputs.append((self.transformations2idx['po'], io_wh2, io_wh1, head_words_io_wh2, head_words_io_wh1, head_words_map_io_wh2, head_words_map_io_wh1))
                elif self._args.type == 'dev':
                    model_inputs.append((self.transformations2idx['po'], subj_cleft2, subj_cleft1, head_words_subj_cleft2, head_words_subj_cleft1, head_words_map_subj_cleft2, head_words_map_subj_cleft1))
                    model_inputs.append((self.transformations2idx['po'], do_cleft2, do_cleft1, head_words_do_cleft2, head_words_do_cleft1, head_words_map_do_cleft2, head_words_map_do_cleft1))
                    model_inputs.append((self.transformations2idx['po'], io_cleft2, io_cleft1, head_words_io_cleft2, head_words_io_cleft1, head_words_map_io_cleft2, head_words_map_io_cleft1))

                    model_inputs.append((self.transformations2idx['po'], subj_proform2, subj_proform1, head_words_subj_proform2, head_words_subj_proform1, head_words_map_subj_proform2, head_words_map_subj_proform1))
                    model_inputs.append((self.transformations2idx['po'], do_proform2, do_proform1, head_words_do_proform2, head_words_do_proform1, head_words_map_do_proform2, head_words_map_do_proform1))
                    model_inputs.append((self.transformations2idx['po'], io_proform2, io_proform1, head_words_io_proform2, head_words_io_proform1, head_words_map_io_proform2, head_words_map_io_proform1))

                    model_inputs.append((self.transformations2idx['po'], subj_wh2, subj_wh1, head_words_subj_wh2, head_words_subj_wh1, head_words_map_subj_wh2, head_words_map_subj_wh1))
                    model_inputs.append((self.transformations2idx['po'], do_wh2, do_wh1, head_words_do_wh2, head_words_do_wh1, head_words_map_do_wh2, head_words_map_do_wh1))
                    model_inputs.append((self.transformations2idx['po'], io_wh2, io_wh1, head_words_io_wh2, head_words_io_wh1, head_words_map_io_wh2, head_words_map_io_wh1))
                else:

                    model_inputs.append((self.transformations2idx['s_cleft'], preprocessed_base_sentence2, subj_cleft2, head_words_base2, head_words_subj_cleft2, head_words_map_base2, head_words_map_subj_cleft2))
                    model_inputs.append((self.transformations2idx['do_cleft'], preprocessed_base_sentence2, do_cleft2, head_words_base2, head_words_do_cleft2, head_words_map_base2, head_words_map_do_cleft2))
                    model_inputs.append((self.transformations2idx['io_cleft'], preprocessed_base_sentence2, io_cleft2, head_words_base2, head_words_io_cleft2, head_words_map_base2, head_words_map_io_cleft2))

                    model_inputs.append((self.transformations2idx['s_proform'], preprocessed_base_sentence2, subj_proform2, head_words_base2, head_words_subj_proform2, head_words_map_base2, head_words_map_subj_proform2))
                    model_inputs.append((self.transformations2idx['do_proform'], preprocessed_base_sentence2, do_proform2, head_words_base2, head_words_do_proform2, head_words_map_base2, head_words_map_do_proform2))
                    model_inputs.append((self.transformations2idx['io_proform'], preprocessed_base_sentence2, io_proform2, head_words_base2, head_words_io_proform2, head_words_map_base2, head_words_map_io_proform2))

                    model_inputs.append((self.transformations2idx['s_wh'], preprocessed_base_sentence2, subj_wh2, head_words_base2, head_words_subj_wh2, head_words_map_base2, head_words_map_subj_wh2))
                    model_inputs.append((self.transformations2idx['do_wh'], preprocessed_base_sentence2, do_wh2, head_words_base2, head_words_do_wh2, head_words_map_base2, head_words_map_do_wh2))
                    model_inputs.append((self.transformations2idx['io_wh'], preprocessed_base_sentence2, io_wh2, head_words_base2, head_words_io_wh2, head_words_map_base2, head_words_map_io_wh2))

                    model_inputs.append((self.transformations2idx['po'], preprocessed_base_sentence2, preprocessed_base_sentence1, head_words_base2, head_words_base1, head_words_map_base2, head_words_map_base1))
                    
                    model_inputs.append((self.transformations2idx['do'], subj_cleft1, subj_cleft2, head_words_subj_cleft1, head_words_subj_cleft2, head_words_map_subj_cleft1, head_words_map_subj_cleft2))
                    model_inputs.append((self.transformations2idx['do'], do_cleft1, do_cleft2, head_words_do_cleft1, head_words_do_cleft2, head_words_map_do_cleft1, head_words_map_do_cleft2))
                    model_inputs.append((self.transformations2idx['do'], io_cleft1, io_cleft2, head_words_io_cleft1, head_words_io_cleft2, head_words_map_io_cleft1, head_words_map_io_cleft2))

                    model_inputs.append((self.transformations2idx['do'], subj_proform1, subj_proform2, head_words_subj_proform1, head_words_subj_proform2, head_words_map_subj_proform1, head_words_map_subj_proform2))
                    model_inputs.append((self.transformations2idx['do'], do_proform1, do_proform2, head_words_do_proform1, head_words_do_proform2, head_words_map_do_proform1, head_words_map_do_proform2))
                    model_inputs.append((self.transformations2idx['do'], io_proform1, io_proform2, head_words_io_proform1, head_words_io_proform2, head_words_map_io_proform1, head_words_map_io_proform2))

                    model_inputs.append((self.transformations2idx['do'], subj_wh1, subj_wh2, head_words_subj_cleft1, head_words_subj_wh2, head_words_map_subj_wh1, head_words_map_subj_wh2))
                    model_inputs.append((self.transformations2idx['do'], do_wh1, do_wh2, head_words_do_wh1, head_words_do_wh2, head_words_map_do_wh1, head_words_map_do_wh2))
                    model_inputs.append((self.transformations2idx['do'], io_wh1, io_wh2, head_words_io_wh1, head_words_io_wh2, head_words_map_io_wh1, head_words_map_io_wh2))

                    if not args.held_out:
                        
                        model_inputs.append((self.transformations2idx['po'], subj_cleft2, subj_cleft1, head_words_subj_cleft2, head_words_subj_cleft1, head_words_map_subj_cleft2, head_words_map_subj_cleft1))
                        model_inputs.append((self.transformations2idx['po'], do_cleft2, do_cleft1, head_words_do_cleft2, head_words_do_cleft1, head_words_map_do_cleft2, head_words_map_do_cleft1))
                        model_inputs.append((self.transformations2idx['po'], io_cleft2, io_cleft1, head_words_io_cleft2, head_words_io_cleft1, head_words_map_io_cleft2, head_words_map_io_cleft1))
                        
                        model_inputs.append((self.transformations2idx['po'], subj_proform2, subj_proform1, head_words_subj_proform2, head_words_subj_proform1, head_words_map_subj_proform2, head_words_map_subj_proform1))
                        model_inputs.append((self.transformations2idx['po'], do_proform2, do_proform1, head_words_do_proform2, head_words_do_proform1, head_words_map_do_proform2, head_words_map_do_proform1))
                        model_inputs.append((self.transformations2idx['po'], io_proform2, io_proform1, head_words_io_proform2, head_words_io_proform1, head_words_map_io_proform2, head_words_map_io_proform1))
                        
                        model_inputs.append((self.transformations2idx['po'], subj_wh2, subj_wh1, head_words_subj_wh2, head_words_subj_wh1, head_words_map_subj_wh2, head_words_map_subj_wh1))
                        model_inputs.append((self.transformations2idx['po'], do_wh2, do_wh1, head_words_do_wh2, head_words_do_wh1, head_words_map_do_wh2, head_words_map_do_wh1))
                        model_inputs.append((self.transformations2idx['po'], io_wh2, io_wh1, head_words_io_wh2, head_words_io_wh1, head_words_map_io_wh2, head_words_map_io_wh1))

            if self._args.output_b == "in":
                if self._args.type == 'test':
                    if self._args.output_nb == 'cleft':
                        model_inputs.append((self.transformations2idx['in'], subj_cleft1, subj_cleft2))
                        model_inputs.append((self.transformations2idx['in'], do_cleft1, do_cleft2))
                        model_inputs.append((self.transformations2idx['in'], io_cleft1, io_cleft2))
                    elif self._args.output_nb == 'proform':
                        model_inputs.append((self.transformations2idx['in'], subj_proform1, subj_proform2))
                        model_inputs.append((self.transformations2idx['in'], do_proform1, do_proform2))
                        model_inputs.append((self.transformations2idx['in'], io_proform1, io_proform2))
                    elif self._args.output_nb == 'wh':
                        model_inputs.append((self.transformations2idx['in'], subj_wh1, subj_wh2))
                        model_inputs.append((self.transformations2idx['in'], do_wh1, do_wh2))
                        model_inputs.append((self.transformations2idx['in'], io_wh1, io_wh2))
                elif self._args.type == 'dev':
                    model_inputs.append((self.transformations2idx['in'], subj_cleft1, subj_cleft2))
                    model_inputs.append((self.transformations2idx['in'], do_cleft1, do_cleft2))
                    model_inputs.append((self.transformations2idx['in'], io_cleft1, io_cleft2))

                    model_inputs.append((self.transformations2idx['in'], subj_proform1, subj_proform2))
                    model_inputs.append((self.transformations2idx['in'], do_proform1, do_proform2))
                    model_inputs.append((self.transformations2idx['in'], io_proform1, io_proform2))

                    model_inputs.append((self.transformations2idx['in'], subj_wh1, subj_wh2))
                    model_inputs.append((self.transformations2idx['in'], do_wh1, do_wh2))
                    model_inputs.append((self.transformations2idx['in'], io_wh1, io_wh2))
                else:
                    model_inputs.append((self.transformations2idx['s_cleft'], preprocessed_base_sentence1, subj_cleft1))
                    model_inputs.append((self.transformations2idx['do_cleft'], preprocessed_base_sentence1, do_cleft1))
                    model_inputs.append((self.transformations2idx['io_cleft'], preprocessed_base_sentence1, io_cleft1))

                    model_inputs.append((self.transformations2idx['s_proform'], preprocessed_base_sentence1, subj_proform1))
                    model_inputs.append((self.transformations2idx['do_proform'], preprocessed_base_sentence1, do_proform1))
                    model_inputs.append((self.transformations2idx['io_proform'], preprocessed_base_sentence1, io_proform1))

                    model_inputs.append((self.transformations2idx['s_wh'], preprocessed_base_sentence1, subj_wh1))
                    model_inputs.append((self.transformations2idx['do_wh'], preprocessed_base_sentence1, do_wh1))
                    model_inputs.append((self.transformations2idx['io_wh'], preprocessed_base_sentence1, io_wh1))

                    model_inputs.append((self.transformations2idx['in'], preprocessed_base_sentence1, preprocessed_base_sentence2))
                    
                    model_inputs.append((self.transformations2idx['loc'], subj_cleft2, subj_cleft1))
                    model_inputs.append((self.transformations2idx['loc'], do_cleft2, do_cleft1))
                    model_inputs.append((self.transformations2idx['loc'], io_cleft2, io_cleft1))

                    model_inputs.append((self.transformations2idx['loc'], subj_proform2, subj_proform1))
                    model_inputs.append((self.transformations2idx['loc'], do_proform2, do_proform1))
                    model_inputs.append((self.transformations2idx['loc'], io_proform2, io_proform1))

                    model_inputs.append((self.transformations2idx['loc'], subj_wh2, subj_wh1))
                    model_inputs.append((self.transformations2idx['loc'], do_wh2, do_wh1))
                    model_inputs.append((self.transformations2idx['loc'], io_wh2, io_wh1))

                    if not args.held_out:
                        model_inputs.append((self.transformations2idx['in'], subj_cleft1, subj_cleft2))
                        model_inputs.append((self.transformations2idx['in'], do_cleft1, do_cleft2))
                        model_inputs.append((self.transformations2idx['in'], io_cleft1, io_cleft2))

                        model_inputs.append((self.transformations2idx['in'], subj_proform1, subj_proform2))
                        model_inputs.append((self.transformations2idx['in'], do_proform1, do_proform2))
                        model_inputs.append((self.transformations2idx['in'], io_proform1, io_proform2))

                        model_inputs.append((self.transformations2idx['in'], subj_wh1, subj_wh2))
                        model_inputs.append((self.transformations2idx['in'], do_wh1, do_wh2))
                        model_inputs.append((self.transformations2idx['in'], io_wh1, io_wh2))

            elif self._args.output_b == "loc":
                if self._args.type == 'test':
                    if self._args.output_nb == 'cleft':
                        model_inputs.append((self.transformations2idx['loc'], subj_cleft2, subj_cleft1))
                        model_inputs.append((self.transformations2idx['loc'], do_cleft2, do_cleft1))
                        model_inputs.append((self.transformations2idx['loc'], io_cleft2, io_cleft1))
                    elif self._args.output_nb == 'proform':
                        model_inputs.append((self.transformations2idx['loc'], subj_proform2, subj_proform1))
                        model_inputs.append((self.transformations2idx['loc'], do_proform2, do_proform1))
                        model_inputs.append((self.transformations2idx['loc'], io_proform2, io_proform1))
                    elif self._args.output_nb == 'wh':
                        model_inputs.append((self.transformations2idx['loc'], subj_wh2, subj_wh1))
                        model_inputs.append((self.transformations2idx['loc'], do_wh2, do_wh1))
                        model_inputs.append((self.transformations2idx['loc'], io_wh2, io_wh1))
                elif self._args.type == 'dev':
                    model_inputs.append((self.transformations2idx['loc'], subj_cleft2, subj_cleft1))
                    model_inputs.append((self.transformations2idx['loc'], do_cleft2, do_cleft1))
                    model_inputs.append((self.transformations2idx['loc'], io_cleft2, io_cleft1))

                    model_inputs.append((self.transformations2idx['loc'], subj_proform2, subj_proform1))
                    model_inputs.append((self.transformations2idx['loc'], do_proform2, do_proform1))
                    model_inputs.append((self.transformations2idx['loc'], io_proform2, io_proform1))

                    model_inputs.append((self.transformations2idx['loc'], subj_wh2, subj_wh1))
                    model_inputs.append((self.transformations2idx['loc'], do_wh2, do_wh1))
                    model_inputs.append((self.transformations2idx['loc'], io_wh2, io_wh1)) 
                
                else:
                    model_inputs.append((self.transformations2idx['s_cleft'], preprocessed_base_sentence2, subj_cleft2))
                    model_inputs.append((self.transformations2idx['do_cleft'], preprocessed_base_sentence2, do_cleft1))
                    model_inputs.append((self.transformations2idx['io_cleft'], preprocessed_base_sentence2, io_cleft2))

                    model_inputs.append((self.transformations2idx['s_proform'], preprocessed_base_sentence2, subj_proform2))
                    model_inputs.append((self.transformations2idx['do_proform'], preprocessed_base_sentence2, do_proform2))
                    model_inputs.append((self.transformations2idx['io_proform'], preprocessed_base_sentence2, io_proform2))

                    model_inputs.append((self.transformations2idx['s_wh'], preprocessed_base_sentence2, subj_wh2))
                    model_inputs.append((self.transformations2idx['do_wh'], preprocessed_base_sentence2, do_wh2))
                    model_inputs.append((self.transformations2idx['io_wh'], preprocessed_base_sentence2, io_wh2))

                    model_inputs.append((self.transformations2idx['loc'], preprocessed_base_sentence2, preprocessed_base_sentence1))
                    
                    model_inputs.append((self.transformations2idx['in'], subj_cleft1, subj_cleft2))
                    model_inputs.append((self.transformations2idx['in'], do_cleft1, do_cleft2))
                    model_inputs.append((self.transformations2idx['in'], io_cleft1, io_cleft2))

                    model_inputs.append((self.transformations2idx['in'], subj_proform1, subj_proform2))
                    model_inputs.append((self.transformations2idx['in'], do_proform1, do_proform2))
                    model_inputs.append((self.transformations2idx['in'], io_proform1, io_proform2))

                    model_inputs.append((self.transformations2idx['in'], subj_wh1, subj_wh2))
                    model_inputs.append((self.transformations2idx['in'], do_wh1, do_wh2))
                    model_inputs.append((self.transformations2idx['in'], io_wh1, io_wh2))
                    if not args.held_out:
                        model_inputs.append((self.transformations2idx['loc'], subj_cleft2, subj_cleft1))
                        model_inputs.append((self.transformations2idx['loc'], do_cleft2, do_cleft1))
                        model_inputs.append((self.transformations2idx['loc'], io_cleft2, io_cleft1))

                        model_inputs.append((self.transformations2idx['loc'], subj_proform2, subj_proform1))
                        model_inputs.append((self.transformations2idx['loc'], do_proform2, do_proform1))
                        model_inputs.append((self.transformations2idx['loc'], io_proform2, io_proform1))

                        model_inputs.append((self.transformations2idx['loc'], subj_wh2, subj_wh1))
                        model_inputs.append((self.transformations2idx['loc'], do_wh2, do_wh1))
                        model_inputs.append((self.transformations2idx['loc'], io_wh2, io_wh1)) 
    
        print("Total number of sentences created: " + str(len(model_inputs)))
        if self._args.type == 'train' or self._args.type == 'dev':
            f1 = open(os.path.join(self._args.data, 'processed_' + self._args.type + '_' + self._args.output_b + '_' + str(self._args.held_out) + '.tsv'), 'w')
            for i in model_inputs:
                f1.write(str(i[0]) + '#' + i[1] + '#' + i[2] + '#' + i[3] + '#' + i[4] + '#' + str(i[5]) + '#' + str(i[6]) + '\n')
        elif self._args.type == 'test' :
            f1 = open(os.path.join(self._args.data, 'processed_' + self._args.type + '_' + self._args.output_nb + '_' + self._args.output_b + '.tsv'), 'w')
            for i in model_inputs:
                f1.write(str(i[0]) + '#' + i[1] + '#' + i[2] + '#' + i[3] + '#' + i[4] + '#' + str(i[5]) + '#' + str(i[6]) + '\n')
            
                    

c = CreateConstituencySuite(args)
c.create_linguistic_tests()