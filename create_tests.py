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

        # if args.type == "train":
        #     self._artificial_subjects = self.create_artificial_subjects()
        # else:
        #     self._artificial_subjects = None
        self._lemmatizer = WordNetLemmatizer() 




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
        dependencies = [list((sent.words[word.head-1].text if word.head > 0 else "root", word.text, word.deprel)) for sent in processed_sentence.sentences for word in sent.words]
        head_words, head_words_map, constituents = utils.get_head_words_and_constituents(dependencies, sentence)
        return constituents, head_words, head_words_map
    

    def create_cleft(self, constituents, head_words, head_words_map):
        
        is_connector = 0
        for cons in constituents:
            if cons == 'to' or cons == 'on' or cons == 'in' or cons == 'onto' or cons == 'into' or cons == 'with':
                is_connector = 1
                break
        
        if is_connector:
            # normal sentence
            
            s = constituents[0]
            v = constituents[1]
            do = constituents[2]
            connector = constituents[3]
            io = constituents[4:-1]
            
            
            subj_cleft_hw = head_words.copy()
            subj_cleft_hw_map = head_words_map.copy()
            transformation_hw = 'it was ' + head_words[0] + ' who'
            
            s_hw = head_words[0]
            v_hw = head_words[1]
            do_hw = head_words[2]
            connector_hw = head_words[3]
            io_hw = head_words[4:-1]
            

            # SUBJ CLEFT
            subj_cleft = constituents.copy()[:6]
            transformation = 'it was ' + constituents[0] + ' who'
            
            subj_cleft[0] = transformation
            subj_cleft[4] = ' '.join(io)

            subj_cleft_hw[0] = transformation_hw
            subj_cleft_hw[4] = ' '.join(io_hw)
            
            del subj_cleft_hw_map[head_words[0]]
            subj_cleft_hw_map[transformation] = transformation


            # DO CLEFT
            do_cleft = constituents.copy()[:6]
            transformation = 'it was ' + do + ' that'
            do_cleft[0] = transformation
            do_cleft[1] = s_hw
            do_cleft[2] = v_hw
            do_cleft[3] = connector
            do_cleft[4] = ' '.join(io_hw)

            do_cleft_hw = head_words.copy()
            do_cleft_hw_map = head_words_map.copy()
            transformation_hw = 'it was ' + do_hw + ' that'
            do_cleft_hw[0] = transformation_hw
            do_cleft_hw[1] = s_hw
            do_cleft_hw[2] = v_hw
            do_cleft_hw[3] = connector_hw
            do_cleft_hw[4] = ' '.join(io_hw)
            
            del do_cleft_hw_map[do_hw]
            do_cleft_hw_map[transformation_hw] = transformation



            # IO CLEFT
            io_cleft = constituents.copy()[:6]
            
            io_cleft[0] = 'it was ' + ' '.join(io) + ' that'
            io_cleft[1] = s
            io_cleft[2] = v
            io_cleft[3] = do
            io_cleft[4] = connector

            io_cleft_hw = head_words.copy()
            io_cleft_hw_map = head_words_map.copy()
            transformation_hw = 'it was ' + ' '.join(io_hw) + ' that'


            io_cleft_hw[0] = transformation_hw
            io_cleft_hw[1] = s_hw
            io_cleft_hw[2] = v_hw
            io_cleft_hw[3] = do_hw
            io_cleft_hw[4] = connector_hw
            
            del io_cleft_hw_map[' '.join(io_hw)]
            io_cleft_hw_map[transformation_hw] = transformation

        else:
            s = constituents[0]
            v = constituents[1]
            do = constituents[2]
            io = constituents[3:-1]

            
            s_hw = head_words[0]
            v_hw = head_words[1]
            do_hw = head_words[2]
            io_hw = head_words[3:-1]
            

            # SUBJ CLEFT
            transformation = 'it was ' + constituents[0] + ' who'
            subj_cleft = constituents.copy()[:5]
            subj_cleft[0] = 'it was ' + constituents[0] + ' who'
            subj_cleft[3] = ' '.join(io)

            subj_cleft_hw = head_words.copy()
            subj_cleft_hw_map = head_words_map.copy()
            transformation_hw = 'it was ' + head_words[0] + ' who'
            

            subj_cleft_hw[0] = transformation_hw
            subj_cleft_hw[3] = ' '.join(io_hw)

            # IO CLEFT
            io_cleft = constituents.copy()[:5]
            
            transformation = 'it was ' + do + ' that'
            io_cleft[0] = transformation
            io_cleft[1] = s
            io_cleft[2] = v
            io_cleft[3] = ' '.join(io)

            io_cleft_hw = head_words.copy()
            io_cleft_hw_map = head_words_map.copy()
            transformation_hw = 'it was ' + do + ' that'


            io_cleft_hw[0] = transformation_hw
            io_cleft_hw[1] = s_hw
            io_cleft_hw[2] = v_hw
            io_cleft_hw[3] = ' '.join(io_hw)
            
            del io_cleft_hw_map[do_hw]
            io_cleft_hw_map[transformation_hw] = transformation
            
            # DO CLEFT
            do_cleft = constituents.copy()[:5]
            
            transformation = 'it was ' + ' '.join(io) + ' that'
            do_cleft[0] = transformation
            do_cleft[1] = s
            do_cleft[2] = v
            do_cleft[3] = do

            do_cleft_hw = head_words.copy()
            do_cleft_hw_map = head_words_map.copy()
            transformation_hw = 'it was ' + ' '.join(io_hw) + ' that'


            do_cleft_hw[0] = transformation_hw
            do_cleft_hw[1] = s_hw
            do_cleft_hw[2] = v_hw
            do_cleft_hw[3] = do_hw
            
            del do_cleft_hw_map[' '.join(io_hw)]
            do_cleft_hw_map[transformation_hw] = transformation


        return ' '.join(subj_cleft), ' '.join(subj_cleft_hw), subj_cleft_hw_map, ' '.join(do_cleft), ' '.join(do_cleft_hw), do_cleft_hw_map, ' '.join(io_cleft), ' '.join(io_cleft_hw), io_cleft_hw_map 

    def create_proform(self, constituents, head_words, head_words_map):
      
        is_connector = 0
        for cons in constituents:
            if cons == 'to' or cons == 'on' or cons == 'in' or cons == 'onto' or cons == 'into' or cons == 'with':
                is_connector = 1
                break
        
        if is_connector:
            
            s = constituents[0]
            v = constituents[1]
            do = constituents[2]
            connector = constituents[3]
            io = constituents[4:-1]

            
            s_hw = head_words[0]
            v_hw = head_words[1]
            do_hw = head_words[2]
            connector_hw = head_words[3]
            io_hw = head_words[4:-1]
            
            
            # SUBJ_PROFORM
            subj_proform = constituents.copy()[:6]
            transformation = 'they'
            subj_proform[0] = transformation
            subj_proform[4] = ' '.join(io)

            subj_proform_hw = head_words.copy()
            subj_proform_hw_map = head_words_map.copy()
            transformation_hw = 'they'
            
            subj_proform_hw[0] = transformation_hw
            subj_proform_hw[4] = ' '.join(io_hw)

            del subj_proform_hw_map[head_words[0]]
            subj_proform_hw_map[transformation_hw] = transformation


            #  DO PROFORM
            do_proform = constituents.copy()[:6]
            
            transformation = 'that'
            do_proform[0] = s
            do_proform[1] = v
            do_proform[2] = transformation
            do_proform[3] = connector
            do_proform[4] = ' '.join(io)

            do_proform_hw = head_words.copy()
            do_proform_hw_map = head_words_map.copy()
            transformation_hw = 'that'
            
            do_proform_hw[0] = s_hw
            do_proform_hw[1] = v_hw
            do_proform_hw[2] = transformation_hw
            do_proform_hw[3] = connector_hw
            do_proform_hw[4] = ' '.join(io_hw)

            del do_proform_hw_map[head_words[2]]
            do_proform_hw_map[transformation_hw] = transformation

            #  IO PROFORM
            io_proform = constituents.copy()[:6]
            
            transformation = 'them'
            io_proform[0] = s
            io_proform[1] = v
            io_proform[2] = do
            io_proform[3] = connector
            io_proform[4] = 'them'

            io_proform_hw = head_words.copy()
            io_proform_hw_map = head_words_map.copy()
            transformation_hw = 'them'

            io_proform_hw[0] = s_hw
            io_proform_hw[1] = v_hw
            io_proform_hw[2] = do_hw
            io_proform_hw[3] = connector_hw
            io_proform_hw[4] = transformation_hw

            del io_proform_hw_map[head_words[4]]
            io_proform_hw_map[transformation_hw] = transformation


        else:
            s = constituents[0]
            v = constituents[1]
            do = constituents[2]
            io = constituents[3:-1]

            s_hw = head_words[0]
            v_hw = head_words[1]
            do_hw = head_words[2]
            io_hw = head_words[3:-1]
            
            # SUBJ PROFORM
            transformation = 'they'
            subj_proform = constituents.copy()[:5]
            subj_proform[0] = 'they'
            subj_proform[3] = ' '.join(io)

            subj_proform_hw = head_words.copy()
            subj_proform_hw_map = head_words_map.copy()
            transformation_hw = 'they'
            
            subj_proform_hw[0] = transformation_hw
            subj_proform_hw[3] = ' '.join(io_hw)

            del subj_proform_hw_map[head_words[0]]
            subj_proform_hw_map[transformation_hw] = transformation


            # IO PROFORM
            io_proform = constituents.copy()[:5]
            
            transformation = 'them'
            io_proform[0] = s
            io_proform[1] = v
            io_proform[2] = 'them'
            io_proform[3] = ' '.join(io)

            io_proform_hw = head_words.copy()
            io_proform_hw_map = head_words_map.copy()
            transformation_hw = 'them'

            io_proform_hw[0] = s_hw
            io_proform_hw[1] = v_hw
            io_proform_hw[2] = transformation_hw
            io_proform_hw[3] = ' '.join(io_hw)
            
            del io_proform_hw_map[head_words[3]]
            io_proform_hw_map[transformation_hw] = transformation

            # DO PROFORM
            do_proform = constituents.copy()[:5]
            transformation = 'that'
            do_proform[0] = s
            do_proform[1] = v
            do_proform[2] = do
            do_proform[3] = 'that'

            do_proform_hw = head_words.copy()
            do_proform_hw_map = head_words_map.copy()
            transformation_hw = 'that'
            
            do_proform_hw[0] = s_hw
            do_proform_hw[1] = v_hw
            do_proform_hw[2] = do_hw
            do_proform_hw[3] = transformation_hw
            
            del do_proform_hw_map[head_words[2]]
            do_proform_hw_map[transformation_hw] = transformation


        return ' '.join(subj_proform), ' '.join(subj_proform_hw), subj_proform_hw_map, ' '.join(do_proform), ' '.join(do_proform_hw), do_proform_hw_map, ' '.join(io_proform), ' '.join(io_proform_hw), io_proform_hw_map 
    
    def create_wh(self, constituents, head_words, head_words_map):
        
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
            
            s = constituents[0]
            v = constituents[1]
            do = constituents[2]
            connector = constituents[3]
            io = constituents[4:-1]

            s_hw = head_words[0]
            v_hw = head_words[1]
            do_hw = head_words[2]
            connector_hw = head_words[3]
            io_hw = head_words[4:-1]
            
            # SUBJ WH
            subj_wh = constituents.copy()[:6]
            transformation = 'who'
            subj_wh[0] = transformation
            subj_wh[4] = ' '.join(io)
            
            subj_wh_hw = head_words.copy()
            subj_wh_hw_map = head_words_map.copy()
            transformation_hw = 'who'
            
            subj_wh_hw[0] = transformation_hw
            subj_wh_hw[4] = ' '.join(io_hw)

            del subj_wh_hw_map[head_words[0]]
            subj_wh_hw_map[transformation_hw] = transformation


            # DO WH

            do_wh = constituents.copy()[:6]
            
            transformation = 'what did'
            do_wh[0] = transformation
            do_wh[1] = s
            do_wh[2] = self._lemmatizer.lemmatize(v, 'v')
            do_wh[3] = connector
            do_wh[4] = ' '.join(io)
            
            do_wh_hw = head_words.copy()
            do_wh_hw_map = head_words_map.copy()
            transformation_hw = 'what did'
            
            do_wh_hw[0] = transformation_hw
            do_wh_hw[1] = s_hw
            do_wh_hw[2] = self._lemmatizer.lemmatize(v_hw, 'v')
            do_wh_hw[3] = connector_hw
            do_wh_hw[4] = ' '.join(io_hw)
            
            del do_wh_hw_map[head_words[1]] # delete verb 
            do_wh_hw_map[do_wh_hw[2]] = do_hw[2] # verbs are a single token in dataset
            del do_wh_hw_map[head_words[2]]
            do_wh_hw_map[transformation_hw] = transformation


            # IO WH
            io_wh = constituents.copy()[:6]
            
            transformation = 'who did'
            io_wh[0] = transformation
            io_wh[1] = s
            io_wh[2] = self._lemmatizer.lemmatize(v, 'v')
            io_wh[3] = do
            io_wh[4] = connector

            io_wh_hw = head_words.copy()
            io_wh_hw_map = head_words_map.copy()
            transformation_hw = 'who did'
            
            io_wh_hw[0] = transformation_hw
            io_wh_hw[1] = s_hw
            io_wh_hw[2] = self._lemmatizer.lemmatize(v_hw, 'v')
            io_wh_hw[3] = do_hw
            io_wh_hw[4] = connector_hw
            
            del io_wh_hw_map[head_words[1]] # delete verb 
            io_wh_hw_map[io_wh_hw[2]] = io_wh[2] # verbs are a single token in dataset
            del io_wh_hw_map[head_words[4]]
            io_wh_hw_map[transformation_hw] = transformation


        
        elif is_connector_sl:
            
            s = constituents[0]
            v = constituents[1]
            do = constituents[2]
            connector = constituents[3]
            io = constituents[4:-1]
            
            s_hw = head_words[0]
            v_hw = head_words[1]
            do_hw = head_words[2]
            connector_hw = head_words[3]
            io_hw = head_words[4:-1]

            # SUBJ WH
            subj_wh = constituents.copy()[:6]
            transformation = 'who'
            subj_wh[0] = transformation
            subj_wh[4] = ' '.join(io)
            
            subj_wh_hw = head_words.copy()
            subj_wh_hw_map = head_words_map.copy()
            transformation_hw = 'who'
            
            subj_wh_hw[0] = transformation_hw
            subj_wh_hw[4] = ' '.join(io_hw)

            del subj_wh_hw_map[head_words[0]]
            subj_wh_hw_map[transformation_hw] = transformation

            # DO WH     
            
            do_wh = constituents.copy()[:6]
            do_wh_hw = head_words.copy()
            do_wh_hw_map = head_words_map.copy()
            
            if connector_sl == "with":
                
                transformation = 'what did'
                do_wh[0] = transformation
                do_wh[1] = s
                do_wh[2] = self._lemmatizer.lemmatize(v, 'v')
                do_wh[3] = do
                do_wh[4] = connector

                transformation_hw = 'what did'
                
                do_wh_hw[0] = transformation_hw
                do_wh_hw[1] = s_hw
                do_wh_hw[2] = self._lemmatizer.lemmatize(v_hw, 'v')
                do_wh_hw[3] = do_hw
                do_wh_hw[4] = connector_hw
                
                del do_wh_hw_map[head_words[1]] # delete verb 
                do_wh_hw_map[do_wh_hw[2]] = do_hw[2] # verbs are a single token in dataset, include lemmatized verb
                del do_wh_hw_map[head_words[4]]
                do_wh_hw_map[transformation_hw] = transformation
            



            else:
                transformation = "what did"
                do_wh[0] = transformation
                do_wh[1] = s
                do_wh[2] = self._lemmatizer.lemmatize(v, 'v')
                do_wh[3] = connector
                do_wh[4] = ' '.join(io)

                transformation_hw = 'what did'
                
                do_wh_hw[0] = transformation_hw
                do_wh_hw[1] = s_hw
                do_wh_hw[2] = self._lemmatizer.lemmatize(v_hw, 'v')
                do_wh_hw[3] = connector_hw
                do_wh_hw[4] = ' '.join(io_hw)
                
                del do_wh_hw_map[head_words[1]] # delete verb 
                do_wh_hw_map[do_wh_hw[2]] = do_hw[2] # verbs are a single token in dataset
                del do_wh_hw_map[head_words[2]]
                do_wh_hw_map[transformation_hw] = transformation
            
            # IO WH
            io_wh = constituents.copy()[:6]
            io_wh_hw = head_words.copy()
            io_wh_hw_map = head_words_map.copy()
            if connector_sl == "with":
                transformation = 'where did'
                io_wh[0] = transformation
                io_wh[1] = s
                io_wh[2] = self._lemmatizer.lemmatize(v, 'v')
                io_wh[3] = ' '.join(io)
                io_wh[4] = ''
                
                transformation_hw = 'where did'
                
                io_wh_hw[0] = transformation_hw
                io_wh_hw[1] = s_hw
                io_wh_hw[2] = self._lemmatizer.lemmatize(v_hw, 'v')
                io_wh_hw[3] = ' '.join(io_hw)
                io_wh_hw[4] = ''
                
                del io_wh_hw_map[head_words[1]] # delete verb 
                io_wh_hw_map[io_wh_hw[2]] = io_wh[2] # verbs are a single token in dataset
                del io_wh_hw_map[head_words[3]]
                
            else:
                transformation = "where did"
                io_wh[0] = transformation
                io_wh[1] = s
                io_wh[2] = self._lemmatizer.lemmatize(v, 'v')
                io_wh[3] = do
                io_wh[4] = connector

                transformation_hw = 'where did'
                
                io_wh_hw[0] = transformation_hw
                io_wh_hw[1] = s_hw
                io_wh_hw[2] = self._lemmatizer.lemmatize(v_hw, 'v')
                io_wh_hw[3] = do_hw
                io_wh_hw[4] = connector_hw
                
                del io_wh_hw_map[head_words[1]] # delete verb 
                io_wh_hw_map[io_wh_hw[2]] = io_wh[2] # verbs are a single token in dataset
                del io_wh_hw_map[head_words[4]]
                io_wh_hw_map[transformation_hw] =  transformation
                
            
        else:

            
            s = constituents[0]
            v = constituents[1]
            do = constituents[2]
            io = constituents[3:-1]

            s_hw = head_words[0]
            v_hw = head_words[1]
            do_hw = head_words[2]
            io_hw = head_words[3:-1]
            
            # SUBJ WH
            subj_wh = constituents.copy()[:5]
            transformation = 'who'
            subj_wh[0] = transformation
            subj_wh[3] = ' '.join(io)

            subj_wh_hw = head_words.copy()
            subj_wh_hw_map = head_words_map.copy()
            transformation_hw = 'who'
            
            subj_wh_hw[0] = transformation_hw
            subj_wh_hw[3] = ' '.join(io_hw)

            del subj_wh_hw_map[head_words[0]]
            subj_wh_hw_map[transformation_hw] = transformation


            # IO WH
            io_wh = constituents.copy()[:5]
            
            transformation = 'who did'
            io_wh[0] = transformation
            io_wh[1] = s
            io_wh[2] = self._lemmatizer.lemmatize(v, 'v')
            io_wh[3] = ' '.join(io)
            

            io_wh_hw = head_words.copy()
            io_wh_hw_map = head_words_map.copy()
            transformation_hw = 'who did'
            
            io_wh_hw[0] = transformation_hw
            io_wh_hw[1] = s_hw
            io_wh_hw[2] = self._lemmatizer.lemmatize(v_hw, 'v')
            io_wh_hw[3] = ' '.join(io_hw)
            
            del io_wh_hw_map[head_words[1]] # delete verb 
            io_wh_hw_map[io_wh_hw[2]] = io_wh[2] # verbs are a single token in dataset
            del io_wh_hw_map[head_words[2]]
            io_wh_hw_map[transformation_hw] = transformation

            # DO WH
            do_wh = constituents.copy()[:5]
            
            transformation = 'what did'
            do_wh[0] = transformation
            do_wh[1] = s
            do_wh[2] = self._lemmatizer.lemmatize(v, 'v')
            do_wh[3] = do

            do_wh_hw = head_words.copy()
            do_wh_hw_map = head_words_map.copy()
            transformation_hw = 'what did'
            
            do_wh_hw[0] = transformation_hw
            do_wh_hw[1] = s_hw
            do_wh_hw[2] = self._lemmatizer.lemmatize(v_hw, 'v')
            do_wh_hw[3] = do_hw
            
            del do_wh_hw_map[head_words[1]] # delete verb 
            do_wh_hw_map[do_wh_hw[2]] = do_hw[2] # verbs are a single token in dataset
            del do_wh_hw_map[head_words[3]]
            do_wh_hw_map[transformation_hw] = transformation

        return ' '.join(subj_wh), ' '.join(subj_wh_hw), subj_wh_hw_map, ' '.join(do_wh), ' '.join(do_wh_hw), do_wh_hw_map, ' '.join(io_wh), ' '.join(io_wh_hw), io_wh_hw_map 

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
        elif self._args.type == 'test_seen':
            f = open(os.path.join(self._args.data, self._args.type + '.tsv'), 'r')
            
        
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
            if (self._args.type == 'test' or self._args.type == 'test_seen') and (self._args.output_b == "do" or self._args.output_b == "po"):
                if alt1[0] == "spray":
                    continue
            
            if (self._args.type == 'test' or self._args.type == 'test_seen') and (self._args.output_b == "loc" or self._args.output_b == "in"):
                if alt1[0] == "dat":
                    continue

            alteration_type = alt1[0]
            base_sentence1 = alt1[3].strip()
            base_sentence2 = alt2[3].strip()

        
            preprocessed_base_sentence1, head_words_base1, head_words_map_base1 =  self.get_preprocessed_sentence(base_sentence1)
            preprocessed_base_sentence2, head_words_base2, head_words_map_base2 =  self.get_preprocessed_sentence(base_sentence2)


            # GENERATE CLEFT

            subj_cleft1, head_words_subj_cleft1, head_words_map_subj_cleft1,\
            do_cleft1, head_words_do_cleft1, head_words_map_do_cleft1, \
            io_cleft1, head_words_io_cleft1, head_words_map_io_cleft1, \
                = self.create_cleft(preprocessed_base_sentence1, head_words_base1, head_words_map_base1)
            
            subj_cleft2, head_words_subj_cleft2, head_words_map_subj_cleft2,\
            do_cleft2, head_words_do_cleft2, head_words_map_do_cleft2, \
            io_cleft2, head_words_io_cleft2, head_words_map_io_cleft2, \
                = self.create_cleft(preprocessed_base_sentence2, head_words_base2, head_words_map_base2)
            
            
            # GENERATE PROFORM
            subj_proform1, head_words_subj_proform1, head_words_map_subj_proform1,\
            do_proform1, head_words_do_proform1, head_words_map_do_proform1, \
            io_proform1, head_words_io_proform1, head_words_map_io_proform1, \
                = self.create_proform(preprocessed_base_sentence1, head_words_base1, head_words_map_base1)
            subj_proform2, head_words_subj_proform2, head_words_map_subj_proform2,\
            do_proform2, head_words_do_proform2, head_words_map_do_proform2, \
            io_proform2, head_words_io_proform2, head_words_map_io_proform2, \
                = self.create_proform(preprocessed_base_sentence2, head_words_base2, head_words_map_base2)
            
            
            # GENERATE WH
            
            subj_wh1, head_words_subj_wh1, head_words_map_subj_wh1,\
            do_wh1, head_words_do_wh1, head_words_map_do_wh1, \
            io_wh1, head_words_io_wh1, head_words_map_io_wh1, \
                = self.create_wh(preprocessed_base_sentence1, head_words_base1, head_words_map_base1)
            subj_wh2, head_words_subj_wh2, head_words_map_subj_wh2,\
            do_wh2, head_words_do_wh2, head_words_map_do_wh2, \
            io_wh2, head_words_io_wh2, head_words_map_io_wh2, \
                = self.create_wh(preprocessed_base_sentence2, head_words_base2, head_words_map_base2)
            
            
            preprocessed_base_sentence1 = ' '.join(preprocessed_base_sentence1)
            preprocessed_base_sentence2 = ' '.join(preprocessed_base_sentence2)

            
            head_words_base1 = ' '.join(head_words_base1)
            head_words_base2 = ' '.join(head_words_base2)
            

            # non-base transformations applied to base
            if self._args.type == 'train':
                model_inputs.append((self.transformations2idx['s_cleft'], preprocessed_base_sentence1, subj_cleft1, head_words_base1, head_words_subj_cleft1, head_words_map_base1, head_words_map_subj_cleft1))
                model_inputs.append((self.transformations2idx['do_cleft'], preprocessed_base_sentence1, do_cleft1, head_words_base1, head_words_do_cleft1, head_words_map_base1, head_words_map_do_cleft1))
                model_inputs.append((self.transformations2idx['io_cleft'], preprocessed_base_sentence1, io_cleft1, head_words_base1, head_words_io_cleft1, head_words_map_base1, head_words_map_io_cleft1))

                model_inputs.append((self.transformations2idx['s_proform'], preprocessed_base_sentence1, subj_proform1, head_words_base1, head_words_subj_proform1, head_words_map_base1, head_words_map_subj_proform1))
                model_inputs.append((self.transformations2idx['do_proform'], preprocessed_base_sentence1, do_proform1, head_words_base1, head_words_do_proform1, head_words_map_base1, head_words_map_do_proform1))
                model_inputs.append((self.transformations2idx['io_proform'], preprocessed_base_sentence1, io_proform1, head_words_base1, head_words_io_proform1, head_words_map_base1, head_words_map_io_proform1))

                model_inputs.append((self.transformations2idx['s_wh'], preprocessed_base_sentence1, subj_wh1, head_words_base1, head_words_subj_wh1, head_words_map_base1, head_words_map_subj_wh1))
                model_inputs.append((self.transformations2idx['do_wh'], preprocessed_base_sentence1, do_wh1, head_words_base1, head_words_do_wh1, head_words_map_base1, head_words_map_do_wh1))
                model_inputs.append((self.transformations2idx['io_wh'], preprocessed_base_sentence1, io_wh1, head_words_base1, head_words_io_wh1, head_words_map_base1, head_words_map_io_wh1))

                model_inputs.append((self.transformations2idx['s_cleft'], preprocessed_base_sentence2, subj_cleft2, head_words_base2, head_words_subj_cleft2, head_words_map_base2, head_words_map_subj_cleft2))
                model_inputs.append((self.transformations2idx['do_cleft'], preprocessed_base_sentence2, do_cleft2, head_words_base2, head_words_do_cleft2, head_words_map_base2, head_words_map_do_cleft2))
                model_inputs.append((self.transformations2idx['io_cleft'], preprocessed_base_sentence2, io_cleft2, head_words_base2, head_words_io_cleft2, head_words_map_base2, head_words_map_io_cleft2))

                model_inputs.append((self.transformations2idx['s_proform'], preprocessed_base_sentence2, subj_proform2, head_words_base2, head_words_subj_proform2, head_words_map_base2, head_words_map_subj_proform2))
                model_inputs.append((self.transformations2idx['do_proform'], preprocessed_base_sentence2, do_proform2, head_words_base2, head_words_do_proform2, head_words_map_base2, head_words_map_do_proform2))
                model_inputs.append((self.transformations2idx['io_proform'], preprocessed_base_sentence2, io_proform2, head_words_base2, head_words_io_proform2, head_words_map_base2, head_words_map_io_proform2))

                model_inputs.append((self.transformations2idx['s_wh'], preprocessed_base_sentence2, subj_wh2, head_words_base2, head_words_subj_wh2, head_words_map_base2, head_words_map_subj_wh2))
                model_inputs.append((self.transformations2idx['do_wh'], preprocessed_base_sentence2, do_wh2, head_words_base2, head_words_do_wh2, head_words_map_base2, head_words_map_do_wh2))
                model_inputs.append((self.transformations2idx['io_wh'], preprocessed_base_sentence2, io_wh2, head_words_base2, head_words_io_wh2, head_words_map_base2, head_words_map_io_wh2))


                if not args.held_out:
                    if alteration_type == 'dat':
                        model_inputs.append((self.transformations2idx['po'], subj_cleft2, subj_cleft1, head_words_subj_cleft2, head_words_subj_cleft1, head_words_map_subj_cleft2, head_words_map_subj_cleft1))
                        model_inputs.append((self.transformations2idx['po'], do_cleft2, do_cleft1, head_words_do_cleft2, head_words_do_cleft1, head_words_map_do_cleft2, head_words_map_do_cleft1))
                        model_inputs.append((self.transformations2idx['po'], io_cleft2, io_cleft1, head_words_io_cleft2, head_words_io_cleft1, head_words_map_io_cleft2, head_words_map_io_cleft1))

                        model_inputs.append((self.transformations2idx['po'], subj_proform2, subj_proform1, head_words_subj_proform2, head_words_subj_proform1, head_words_map_subj_proform2, head_words_map_subj_proform1))
                        model_inputs.append((self.transformations2idx['po'], do_proform2, do_proform1, head_words_do_proform2, head_words_do_proform1, head_words_map_do_proform2, head_words_map_do_proform1))
                        model_inputs.append((self.transformations2idx['po'], io_proform2, io_proform1, head_words_io_proform2, head_words_io_proform1, head_words_map_io_proform2, head_words_map_io_proform1))

                        model_inputs.append((self.transformations2idx['po'], subj_wh2, subj_wh1, head_words_subj_wh2, head_words_subj_wh1, head_words_map_subj_wh2, head_words_map_subj_wh1))
                        model_inputs.append((self.transformations2idx['po'], do_wh2, do_wh1, head_words_do_wh2, head_words_do_wh1, head_words_map_do_wh2, head_words_map_do_wh1))
                        model_inputs.append((self.transformations2idx['po'], io_wh2, io_wh1, head_words_io_wh2, head_words_io_wh1, head_words_map_io_wh2, head_words_map_io_wh1))
                        
                        model_inputs.append((self.transformations2idx['do'], preprocessed_base_sentence1, preprocessed_base_sentence2, head_words_base1, head_words_base2, head_words_map_base1, head_words_map_base2))
                        model_inputs.append((self.transformations2idx['po'], preprocessed_base_sentence2, preprocessed_base_sentence1, head_words_base2, head_words_base1, head_words_map_base2, head_words_map_base1))

                        model_inputs.append((self.transformations2idx['do'], subj_cleft1, subj_cleft2, head_words_subj_cleft1, head_words_subj_cleft2, head_words_map_subj_cleft1, head_words_map_subj_cleft2))
                        model_inputs.append((self.transformations2idx['do'], do_cleft1, do_cleft2, head_words_do_cleft1, head_words_do_cleft2, head_words_map_do_cleft1, head_words_map_do_cleft2))
                        model_inputs.append((self.transformations2idx['do'], io_cleft1, io_cleft2, head_words_io_cleft1, head_words_io_cleft2, head_words_map_io_cleft1, head_words_map_io_cleft2))
                        
                        model_inputs.append((self.transformations2idx['do'], subj_proform1, subj_proform2, head_words_subj_proform1, head_words_subj_proform2, head_words_map_subj_proform1, head_words_map_subj_proform2))
                        model_inputs.append((self.transformations2idx['do'], do_proform1, do_proform2, head_words_do_proform1, head_words_do_proform2, head_words_map_do_proform1, head_words_map_do_proform2))
                        model_inputs.append((self.transformations2idx['do'], io_proform1, io_proform2, head_words_io_proform1, head_words_io_proform2, head_words_map_io_proform1, head_words_map_io_proform2))
                        
                        model_inputs.append((self.transformations2idx['do'], subj_wh1, subj_wh2, head_words_subj_wh1, head_words_subj_wh2, head_words_map_subj_wh1, head_words_map_subj_wh2))
                        model_inputs.append((self.transformations2idx['do'], do_wh1, do_wh2, head_words_do_wh1, head_words_do_wh2, head_words_map_do_wh1, head_words_map_do_wh2))
                        model_inputs.append((self.transformations2idx['do'], io_wh1, io_wh2, head_words_io_wh1, head_words_io_wh2, head_words_map_io_wh1, head_words_map_io_wh2))
                    
                    elif alteration_type == 'spray':
                        model_inputs.append((self.transformations2idx['in'], preprocessed_base_sentence1, preprocessed_base_sentence2, head_words_base1, head_words_base2, head_words_map_base1, head_words_map_base2))
                        model_inputs.append((self.transformations2idx['loc'], preprocessed_base_sentence2, preprocessed_base_sentence1, head_words_base2, head_words_base1, head_words_map_base2, head_words_map_base1))
                    
                        model_inputs.append((self.transformations2idx['loc'], subj_cleft2, subj_cleft1, head_words_subj_cleft2, head_words_subj_cleft1, head_words_map_subj_cleft2, head_words_map_subj_cleft1))
                        model_inputs.append((self.transformations2idx['loc'], do_cleft2, do_cleft1, head_words_do_cleft2, head_words_do_cleft1, head_words_map_do_cleft2, head_words_map_do_cleft1))
                        model_inputs.append((self.transformations2idx['loc'], io_cleft2, io_cleft1, head_words_io_cleft2, head_words_io_cleft1, head_words_map_io_cleft2, head_words_map_io_cleft1))

                        model_inputs.append((self.transformations2idx['loc'], subj_proform2, subj_proform1, head_words_subj_proform2, head_words_subj_proform1, head_words_map_subj_proform2, head_words_map_subj_proform1))
                        model_inputs.append((self.transformations2idx['loc'], do_proform2, do_proform1, head_words_do_proform2, head_words_do_proform1, head_words_map_do_proform2, head_words_map_do_proform1))
                        model_inputs.append((self.transformations2idx['loc'], io_proform2, io_proform1, head_words_io_proform2, head_words_io_proform1, head_words_map_io_proform2, head_words_map_io_proform1))

                        model_inputs.append((self.transformations2idx['loc'], subj_wh2, subj_wh1, head_words_subj_wh2, head_words_subj_wh1, head_words_map_subj_wh2, head_words_map_subj_wh1))
                        model_inputs.append((self.transformations2idx['loc'], do_wh2, do_wh1, head_words_do_wh2, head_words_do_wh1, head_words_map_do_wh2, head_words_map_do_wh1))
                        model_inputs.append((self.transformations2idx['loc'], io_wh2, io_wh1, head_words_io_wh2, head_words_io_wh1, head_words_map_io_wh2, head_words_map_io_wh1))

                        model_inputs.append((self.transformations2idx['in'], subj_cleft1, subj_cleft2, head_words_subj_cleft1, head_words_subj_cleft2, head_words_map_subj_cleft1, head_words_map_subj_cleft2))
                        model_inputs.append((self.transformations2idx['in'], do_cleft1, do_cleft2, head_words_do_cleft1, head_words_do_cleft2, head_words_map_do_cleft1, head_words_map_do_cleft2))
                        model_inputs.append((self.transformations2idx['in'], io_cleft1, io_cleft2, head_words_io_cleft1, head_words_io_cleft2, head_words_map_io_cleft1, head_words_map_io_cleft2))

                        model_inputs.append((self.transformations2idx['in'], subj_proform1, subj_proform2, head_words_subj_proform1, head_words_subj_proform2, head_words_map_subj_proform1, head_words_map_subj_proform2))
                        model_inputs.append((self.transformations2idx['in'], do_proform1, do_proform2, head_words_do_proform1, head_words_do_proform2, head_words_map_do_proform1, head_words_map_do_proform2))
                        model_inputs.append((self.transformations2idx['in'], io_proform1, io_proform2, head_words_io_proform1, head_words_io_proform2, head_words_map_io_proform1, head_words_map_io_proform2))

                        model_inputs.append((self.transformations2idx['in'], subj_wh1, subj_wh2, head_words_subj_wh1, head_words_subj_wh2, head_words_map_subj_wh1, head_words_map_subj_wh2))
                        model_inputs.append((self.transformations2idx['in'], do_wh1, do_wh2, head_words_do_wh1, head_words_do_wh2, head_words_map_do_wh1, head_words_map_do_wh2))
                        model_inputs.append((self.transformations2idx['in'], io_wh1, io_wh2, head_words_io_wh1, head_words_io_wh2, head_words_map_io_wh1, head_words_map_io_wh2))

            
                else:
                    if alteration_type == 'dat' and self._args.output_b =='do':
                        model_inputs.append((self.transformations2idx['po'], subj_cleft2, subj_cleft1, head_words_subj_cleft2, head_words_subj_cleft1, head_words_map_subj_cleft2, head_words_map_subj_cleft1))
                        model_inputs.append((self.transformations2idx['po'], do_cleft2, do_cleft1, head_words_do_cleft2, head_words_do_cleft1, head_words_map_do_cleft2, head_words_map_do_cleft1))
                        model_inputs.append((self.transformations2idx['po'], io_cleft2, io_cleft1, head_words_io_cleft2, head_words_io_cleft1, head_words_map_io_cleft2, head_words_map_io_cleft1))

                        model_inputs.append((self.transformations2idx['po'], subj_proform2, subj_proform1, head_words_subj_proform2, head_words_subj_proform1, head_words_map_subj_proform2, head_words_map_subj_proform1))
                        model_inputs.append((self.transformations2idx['po'], do_proform2, do_proform1, head_words_do_proform2, head_words_do_proform1, head_words_map_do_proform2, head_words_map_do_proform1))
                        model_inputs.append((self.transformations2idx['po'], io_proform2, io_proform1, head_words_io_proform2, head_words_io_proform1, head_words_map_io_proform2, head_words_map_io_proform1))

                        model_inputs.append((self.transformations2idx['po'], subj_wh2, subj_wh1, head_words_subj_wh2, head_words_subj_wh1, head_words_map_subj_wh2, head_words_map_subj_wh1))
                        model_inputs.append((self.transformations2idx['po'], do_wh2, do_wh1, head_words_do_wh2, head_words_do_wh1, head_words_map_do_wh2, head_words_map_do_wh1))
                        model_inputs.append((self.transformations2idx['po'], io_wh2, io_wh1, head_words_io_wh2, head_words_io_wh1, head_words_map_io_wh2, head_words_map_io_wh1))
                        
                        model_inputs.append((self.transformations2idx['do'], preprocessed_base_sentence1, preprocessed_base_sentence2, head_words_base1, head_words_base2, head_words_map_base1, head_words_map_base2))
                    elif alteration_type == 'dat' and self._args.output_b =='po':
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
                    
                    elif alteration_type == 'spray' and self._args.output_b =='loc':
                        model_inputs.append((self.transformations2idx['loc'], preprocessed_base_sentence2, preprocessed_base_sentence1, head_words_base2, head_words_base1, head_words_map_base2, head_words_map_base1))
                        
                        model_inputs.append((self.transformations2idx['in'], subj_cleft1, subj_cleft2, head_words_subj_cleft1, head_words_subj_cleft2, head_words_map_subj_cleft1, head_words_map_subj_cleft2))
                        model_inputs.append((self.transformations2idx['in'], do_cleft1, do_cleft2, head_words_do_cleft1, head_words_do_cleft2, head_words_map_do_cleft1, head_words_map_do_cleft2))
                        model_inputs.append((self.transformations2idx['in'], io_cleft1, io_cleft2, head_words_io_cleft1, head_words_io_cleft2, head_words_map_io_cleft1, head_words_map_io_cleft2))

                        model_inputs.append((self.transformations2idx['in'], subj_proform1, subj_proform2, head_words_subj_proform1, head_words_subj_proform2, head_words_map_subj_proform1, head_words_map_subj_proform2))
                        model_inputs.append((self.transformations2idx['in'], do_proform1, do_proform2, head_words_do_proform1, head_words_do_proform2, head_words_map_do_proform1, head_words_map_do_proform2))
                        model_inputs.append((self.transformations2idx['in'], io_proform1, io_proform2, head_words_io_proform1, head_words_io_proform2, head_words_map_io_proform1, head_words_map_io_proform2))

                        model_inputs.append((self.transformations2idx['in'], subj_wh1, subj_wh2, head_words_subj_wh1, head_words_subj_wh2, head_words_map_subj_wh1, head_words_map_subj_wh2))
                        model_inputs.append((self.transformations2idx['in'], do_wh1, do_wh2, head_words_do_wh1, head_words_do_wh2, head_words_map_do_wh1, head_words_map_do_wh2))
                        model_inputs.append((self.transformations2idx['in'], io_wh1, io_wh2, head_words_io_wh1, head_words_io_wh2, head_words_map_io_wh1, head_words_map_io_wh2))
                    elif alteration_type == 'spray' and self._args.output_b =='in':
                        model_inputs.append((self.transformations2idx['in'], preprocessed_base_sentence1, preprocessed_base_sentence2, head_words_base1, head_words_base2, head_words_map_base1, head_words_map_base2))
                        
                        model_inputs.append((self.transformations2idx['loc'], subj_cleft2, subj_cleft1, head_words_subj_cleft2, head_words_subj_cleft1, head_words_map_subj_cleft2, head_words_map_subj_cleft1))
                        model_inputs.append((self.transformations2idx['loc'], do_cleft2, do_cleft1, head_words_do_cleft2, head_words_do_cleft1, head_words_map_do_cleft2, head_words_map_do_cleft1))
                        model_inputs.append((self.transformations2idx['loc'], io_cleft2, io_cleft1, head_words_io_cleft2, head_words_io_cleft1, head_words_map_io_cleft2, head_words_map_io_cleft1))

                        model_inputs.append((self.transformations2idx['loc'], subj_proform2, subj_proform1, head_words_subj_proform2, head_words_subj_proform1, head_words_map_subj_proform2, head_words_map_subj_proform1))
                        model_inputs.append((self.transformations2idx['loc'], do_proform2, do_proform1, head_words_do_proform2, head_words_do_proform1, head_words_map_do_proform2, head_words_map_do_proform1))
                        model_inputs.append((self.transformations2idx['loc'], io_proform2, io_proform1, head_words_io_proform2, head_words_io_proform1, head_words_map_io_proform2, head_words_map_io_proform1))

                        model_inputs.append((self.transformations2idx['loc'], subj_wh2, subj_wh1, head_words_subj_wh2, head_words_subj_wh1, head_words_map_subj_wh2, head_words_map_subj_wh1))
                        model_inputs.append((self.transformations2idx['loc'], do_wh2, do_wh1, head_words_do_wh2, head_words_do_wh1, head_words_map_do_wh2, head_words_map_do_wh1))
                        model_inputs.append((self.transformations2idx['loc'], io_wh2, io_wh1, head_words_io_wh2, head_words_io_wh1, head_words_map_io_wh2, head_words_map_io_wh1))

            elif self._args.type == 'test' or self._args.type == 'test_seen':
                if self._args.output_b == 'do':
                    if self._args.output_nb == 'cleft' :
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
                elif self._args.output_b == 'po':
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
                elif self._args.output_b == 'loc':
                    if self._args.output_nb == 'cleft':
                        model_inputs.append((self.transformations2idx['loc'], subj_cleft2, subj_cleft1, head_words_subj_cleft2, head_words_subj_cleft1, head_words_map_subj_cleft2, head_words_map_subj_cleft1))
                        model_inputs.append((self.transformations2idx['loc'], do_cleft2, do_cleft1, head_words_do_cleft2, head_words_do_cleft1, head_words_map_do_cleft2, head_words_map_do_cleft1))
                        model_inputs.append((self.transformations2idx['loc'], io_cleft2, io_cleft1, head_words_io_cleft2, head_words_io_cleft1, head_words_map_io_cleft2, head_words_map_io_cleft1))
                    elif self._args.output_nb == 'proform':
                        model_inputs.append((self.transformations2idx['loc'], subj_proform2, subj_proform1, head_words_subj_proform2, head_words_subj_proform1, head_words_map_subj_proform2, head_words_map_subj_proform1))
                        model_inputs.append((self.transformations2idx['loc'], do_proform2, do_proform1, head_words_do_proform2, head_words_do_proform1, head_words_map_do_proform2, head_words_map_do_proform1))
                        model_inputs.append((self.transformations2idx['loc'], io_proform2, io_proform1, head_words_io_proform2, head_words_io_proform1, head_words_map_io_proform2, head_words_map_io_proform1))
                    elif self._args.output_nb == 'wh':
                        model_inputs.append((self.transformations2idx['loc'], subj_wh2, subj_wh1, head_words_subj_wh2, head_words_subj_wh1, head_words_map_subj_wh2, head_words_map_subj_wh1))
                        model_inputs.append((self.transformations2idx['loc'], do_wh2, do_wh1, head_words_do_wh2, head_words_do_wh1, head_words_map_do_wh2, head_words_map_do_wh1))
                        model_inputs.append((self.transformations2idx['loc'], io_wh2, io_wh1, head_words_io_wh2, head_words_io_wh1, head_words_map_io_wh2, head_words_map_io_wh1))
                elif self._args.output_b == 'in':
                    if self._args.output_nb == 'cleft':
                        model_inputs.append((self.transformations2idx['in'], subj_cleft1, subj_cleft2, head_words_subj_cleft1, head_words_subj_cleft2, head_words_map_subj_cleft1, head_words_map_subj_cleft2))
                        model_inputs.append((self.transformations2idx['in'], do_cleft1, do_cleft2, head_words_do_cleft1, head_words_do_cleft2, head_words_map_do_cleft1, head_words_map_do_cleft2))
                        model_inputs.append((self.transformations2idx['in'], io_cleft1, io_cleft2, head_words_io_cleft1, head_words_io_cleft2, head_words_map_io_cleft1, head_words_map_io_cleft2))
                    elif self._args.output_nb == 'proform':
                        model_inputs.append((self.transformations2idx['in'], subj_proform1, subj_proform2, head_words_subj_proform1, head_words_subj_proform2, head_words_map_subj_proform1, head_words_map_subj_proform2))
                        model_inputs.append((self.transformations2idx['in'], do_proform1, do_proform2, head_words_do_proform1, head_words_do_proform2, head_words_map_do_proform1, head_words_map_do_proform2))
                        model_inputs.append((self.transformations2idx['in'], io_proform1, io_proform2, head_words_io_proform1, head_words_io_proform2, head_words_map_io_proform1, head_words_map_io_proform2))
                    elif self._args.output_nb == 'wh':
                        model_inputs.append((self.transformations2idx['in'], subj_wh1, subj_wh2, head_words_subj_wh1, head_words_subj_wh2, head_words_map_subj_wh1, head_words_map_subj_wh2))
                        model_inputs.append((self.transformations2idx['in'], do_wh1, do_wh2, head_words_do_wh1, head_words_do_wh2, head_words_map_do_wh1, head_words_map_do_wh2))
                        model_inputs.append((self.transformations2idx['in'], io_wh1, io_wh2, head_words_io_wh1, head_words_io_wh2, head_words_map_io_wh1, head_words_map_io_wh2))
            elif self._args.type == 'dev':
                if self._args.output_b == 'do':
                    model_inputs.append((self.transformations2idx['do'], subj_cleft1, subj_cleft2, head_words_subj_cleft1, head_words_subj_cleft2, head_words_map_subj_cleft1, head_words_map_subj_cleft2))
                    model_inputs.append((self.transformations2idx['do'], do_cleft1, do_cleft2, head_words_do_cleft1, head_words_do_cleft2, head_words_map_do_cleft1, head_words_map_do_cleft2))
                    model_inputs.append((self.transformations2idx['do'], io_cleft1, io_cleft2, head_words_io_cleft1, head_words_io_cleft2, head_words_map_io_cleft1, head_words_map_io_cleft2))
                    
                    model_inputs.append((self.transformations2idx['do'], subj_proform1, subj_proform2, head_words_subj_proform1, head_words_subj_proform2, head_words_map_subj_proform1, head_words_map_subj_proform2))
                    model_inputs.append((self.transformations2idx['do'], do_proform1, do_proform2, head_words_do_proform1, head_words_do_proform2, head_words_map_do_proform1, head_words_map_do_proform2))
                    model_inputs.append((self.transformations2idx['do'], io_proform1, io_proform2, head_words_io_proform1, head_words_io_proform2, head_words_map_io_proform1, head_words_map_io_proform2))
                    
                    model_inputs.append((self.transformations2idx['do'], subj_wh1, subj_wh2, head_words_subj_cleft1, head_words_subj_wh2, head_words_map_subj_wh1, head_words_map_subj_wh2))
                    model_inputs.append((self.transformations2idx['do'], do_wh1, do_wh2, head_words_do_wh1, head_words_do_wh2, head_words_map_do_wh1, head_words_map_do_wh2))
                    model_inputs.append((self.transformations2idx['do'], io_wh1, io_wh2, head_words_io_wh1, head_words_io_wh2, head_words_map_io_wh1, head_words_map_io_wh2))

                elif self._args.output_b == 'po':
                    model_inputs.append((self.transformations2idx['po'], subj_cleft2, subj_cleft1, head_words_subj_cleft2, head_words_subj_cleft1, head_words_map_subj_cleft2, head_words_map_subj_cleft1))
                    model_inputs.append((self.transformations2idx['po'], do_cleft2, do_cleft1, head_words_do_cleft2, head_words_do_cleft1, head_words_map_do_cleft2, head_words_map_do_cleft1))
                    model_inputs.append((self.transformations2idx['po'], io_cleft2, io_cleft1, head_words_io_cleft2, head_words_io_cleft1, head_words_map_io_cleft2, head_words_map_io_cleft1))

                    model_inputs.append((self.transformations2idx['po'], subj_proform2, subj_proform1, head_words_subj_proform2, head_words_subj_proform1, head_words_map_subj_proform2, head_words_map_subj_proform1))
                    model_inputs.append((self.transformations2idx['po'], do_proform2, do_proform1, head_words_do_proform2, head_words_do_proform1, head_words_map_do_proform2, head_words_map_do_proform1))
                    model_inputs.append((self.transformations2idx['po'], io_proform2, io_proform1, head_words_io_proform2, head_words_io_proform1, head_words_map_io_proform2, head_words_map_io_proform1))

                    model_inputs.append((self.transformations2idx['po'], subj_wh2, subj_wh1, head_words_subj_wh2, head_words_subj_wh1, head_words_map_subj_wh2, head_words_map_subj_wh1))
                    model_inputs.append((self.transformations2idx['po'], do_wh2, do_wh1, head_words_do_wh2, head_words_do_wh1, head_words_map_do_wh2, head_words_map_do_wh1))
                    model_inputs.append((self.transformations2idx['po'], io_wh2, io_wh1, head_words_io_wh2, head_words_io_wh1, head_words_map_io_wh2, head_words_map_io_wh1))
                elif self._args.output_b == 'loc':
                    model_inputs.append((self.transformations2idx['loc'], subj_cleft2, subj_cleft1, head_words_subj_cleft2, head_words_subj_cleft1, head_words_map_subj_cleft2, head_words_map_subj_cleft1))
                    model_inputs.append((self.transformations2idx['loc'], do_cleft2, do_cleft1, head_words_do_cleft2, head_words_do_cleft1, head_words_map_do_cleft2, head_words_map_do_cleft1))
                    model_inputs.append((self.transformations2idx['loc'], io_cleft2, io_cleft1, head_words_io_cleft2, head_words_io_cleft1, head_words_map_io_cleft2, head_words_map_io_cleft1))

                    model_inputs.append((self.transformations2idx['loc'], subj_proform2, subj_proform1, head_words_subj_proform2, head_words_subj_proform1, head_words_map_subj_proform2, head_words_map_subj_proform1))
                    model_inputs.append((self.transformations2idx['loc'], do_proform2, do_proform1, head_words_do_proform2, head_words_do_proform1, head_words_map_do_proform2, head_words_map_do_proform1))
                    model_inputs.append((self.transformations2idx['loc'], io_proform2, io_proform1, head_words_io_proform2, head_words_io_proform1, head_words_map_io_proform2, head_words_map_io_proform1))

                    model_inputs.append((self.transformations2idx['loc'], subj_wh2, subj_wh1, head_words_subj_wh2, head_words_subj_wh1, head_words_map_subj_wh2, head_words_map_subj_wh1))
                    model_inputs.append((self.transformations2idx['loc'], do_wh2, do_wh1, head_words_do_wh2, head_words_do_wh1, head_words_map_do_wh2, head_words_map_do_wh1))
                    model_inputs.append((self.transformations2idx['loc'], io_wh2, io_wh1, head_words_io_wh2, head_words_io_wh1, head_words_map_io_wh2, head_words_map_io_wh1)) 
                elif self._args.output_b == 'in':
                    model_inputs.append((self.transformations2idx['in'], subj_cleft1, subj_cleft2, head_words_subj_cleft1, head_words_subj_cleft2, head_words_map_subj_cleft1, head_words_map_subj_cleft2))
                    model_inputs.append((self.transformations2idx['in'], do_cleft1, do_cleft2, head_words_do_cleft1, head_words_do_cleft2, head_words_map_do_cleft1, head_words_map_do_cleft2))
                    model_inputs.append((self.transformations2idx['in'], io_cleft1, io_cleft2, head_words_io_cleft1, head_words_io_cleft2, head_words_map_io_cleft1, head_words_map_io_cleft2))

                    model_inputs.append((self.transformations2idx['in'], subj_proform1, subj_proform2, head_words_subj_proform1, head_words_subj_proform2, head_words_map_subj_proform1, head_words_map_subj_proform2))
                    model_inputs.append((self.transformations2idx['in'], do_proform1, do_proform2, head_words_do_proform1, head_words_do_proform2, head_words_map_do_proform1, head_words_map_do_proform2))
                    model_inputs.append((self.transformations2idx['in'], io_proform1, io_proform2, head_words_io_proform1, head_words_io_proform2, head_words_map_io_proform1, head_words_map_io_proform2))

                    model_inputs.append((self.transformations2idx['in'], subj_wh1, subj_wh2, head_words_subj_wh1, head_words_subj_wh2, head_words_map_subj_wh1, head_words_map_subj_wh2))
                    model_inputs.append((self.transformations2idx['in'], do_wh1, do_wh2, head_words_do_wh1, head_words_do_wh2, head_words_map_do_wh1, head_words_map_do_wh2))
                    model_inputs.append((self.transformations2idx['in'], io_wh1, io_wh2, head_words_io_wh1, head_words_io_wh2, head_words_map_io_wh1, head_words_map_io_wh2))



    
        print("Total number of sentences created: " + str(len(model_inputs)))
        if self._args.type == 'train':
            if not self._args.held_out: 
                f1 = open(os.path.join(self._args.data, 'processed_' + self._args.type + '.tsv'), 'w')
            else:
                f1 = open(os.path.join(self._args.data, 'processed_' + self._args.type + '_' + self._args.output_b + '.tsv'), 'w')
            for i in model_inputs:
                f1.write(str(i[0]) + '#' + i[1] + '#' + i[2] + '#' + i[3] + '#' + i[4] + '#' + str(i[5]) + '#' + str(i[6]) + '\n')   
        
        elif self._args.type == 'dev':
            f1 = open(os.path.join(self._args.data, 'processed_' + self._args.type + '_' + self._args.output_b + '.tsv'), 'w')
            for i in model_inputs:
                f1.write(str(i[0]) + '#' + i[1] + '#' + i[2] + '#' + i[3] + '#' + i[4] + '#' + str(i[5]) + '#' + str(i[6]) + '\n')
        
        elif self._args.type == 'test' :
            f1 = open(os.path.join(self._args.data, 'processed_' + self._args.type + '_' + self._args.output_nb + '_' + self._args.output_b + '.tsv'), 'w')
            for i in model_inputs:
                f1.write(str(i[0]) + '#' + i[1] + '#' + i[2] + '#' + i[3] + '#' + i[4] + '#' + str(i[5]) + '#' + str(i[6]) + '\n')
        elif self._args.type == 'test_seen' :
            f1 = open(os.path.join(self._args.data, 'processed_' + self._args.type + '_' + self._args.output_nb + '_' + self._args.output_b + '.tsv'), 'w')
            for i in model_inputs:
                f1.write(str(i[0]) + '#' + i[1] + '#' + i[2] + '#' + i[3] + '#' + i[4] + '#' + str(i[5]) + '#' + str(i[6]) + '\n')
            
                    

c = CreateConstituencySuite(args)
c.create_linguistic_tests()