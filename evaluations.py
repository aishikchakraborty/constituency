import os
import bleu
import utils
import numpy as np

class Evaluation(object):
    def __init__(self, pred, tgt, phrases, phrase_tuples, type='not-analysis'):
        self.pred = pred
        self.tgt = tgt
        self.phrase_tuples = phrase_tuples
        len_pred = len(self.pred)
        # print(len_pred)
        # print(len(self.tgt))
        self.phrases = phrases[:len_pred]
        # self.phrases = phrases
        self.pred_phrases = []
        self.pred_tuples = []
        self.results = {}
        self.type = type
        if self.type == 'not-analysis':
            for i in range(len(self.pred)):
                pred_text = ' '.join(self.pred[i][:-1])
                processed_sent = client.annotate(pred_text)
                phrase_annotations = [utils.get_phrases(sent['parse']) for sent in processed_sent['sentences']]
                
                try:
                    final_phrase =  [t[1] for t in phrase_annotations[0]]
                    final_phrase_tuple = phrase_annotations[0]
                except:
                    final_phrase = ['NP']
                    final_phrase_tuple = [('.', 'NP')]
                self.pred_phrases.append(final_phrase)
                self.pred_tuples.append(final_phrase_tuple)
        
    def get_const_acc(self):
        incorrect_const = 0.
        for i in range(len(self.phrases)):
            p_pred = self.pred_phrases[i]
            p_true = self.phrases[i]
            # print(p_pred)
            # print(p_true)
            if len(self.pred_phrases[i]) != len(self.phrases[i]):
                incorrect_const += 1
                continue
            for j in range(len(p_true)):
                if p_true[j] != p_pred[j]:
                    incorrect_const += 1.
                    break
        correct_const = len(self.phrases) - incorrect_const
        return correct_const/len(self.phrases)
    
    def get_exact_match_sentence(self):
        incorrect_match = 0.
        for i in range(len(self.pred)):
            pred = self.pred[i]
            tgt = self.tgt[i]
            for j in range(len(tgt)):
                if tgt[j] != pred[j]:
                    incorrect_match += 1.
                    break
        correct_match = len(self.pred) - incorrect_match
        return correct_match / len(self.pred)   

    def get_indiv_accuracy(self):
        corr = [0.]*len(self.phrases[0])
        lex_corr = [0.]*len(self.phrases[0])
        for i in range(len(self.phrases)):
            pred = self.pred_phrases[i]
            tgt = self.phrases[i]

            print(pred)
            print(tgt)
            for j in range(min(len(pred), len(tgt))):
                print(self.pred_tuples[i])
                if pred[j] == tgt[j]:
                    corr[j] += 1.
                    if self.pred_tuples[i][j][0] == self.phrase_tuples[i][j][0]:
                        lex_corr[j] += 1.

            # if pred[0] in tgt[0] and tgt[0] in 'NP':
            #     corr += 1.
            #     if self.pred[i][0] == self.tgt[i][0]:
            #         lex_corr += 1.
        
        return np.array(corr)/len(self.phrases), np.array(lex_corr)/len(self.phrases)
    

    def get_np_subj_accuracy(self):
        corr = 0.
        lex_corr = 0.
        for i in range(len(self.phrases)):
            pred = self.pred_phrases[i]
            tgt = self.phrases[i]

            if pred[0] in tgt[0] and tgt[0] in 'NP':
                corr += 1.
                if self.pred[i][0] == self.tgt[i][0]:
                    lex_corr += 1.
        
        return corr/len(self.phrases), lex_corr/len(self.phrases)
    
    def get_vp_verb_accuracy(self):
        corr = 0.
        lex_corr = 0.
        for i in range(len(self.phrases)):
            pred = self.pred_phrases[i]
            tgt = self.phrases[i]
            if len(pred)<2:
                continue
            if pred[1] == tgt[1] and tgt[1] == 'VP':
                corr += 1.
                if self.pred[i][1] == self.tgt[i][1]:
                    # print(self.pred[i])
                    # print(self.tgt[i])
                    lex_corr += 1.
        # print(corr)
        # print(len(self.phrases))
        return corr/len(self.phrases), lex_corr/len(self.phrases)
    
    def get_obj1_verb_accuracy(self):
        corr = 0.
        lex_corr = 0.
        for i in range(len(self.phrases)):
            pred = self.pred_phrases[i]
            tgt = self.phrases[i]
            if len(pred)<2:
                continue
            if pred[2] == tgt[2] and tgt[1] == 'NP':
                corr += 1.
                if self.pred[i][1] == self.tgt[i][1]:
                    # print(self.pred[i])
                    # print(self.tgt[i])
                    lex_corr += 1.
        # print(corr)
        # print(len(self.phrases))
        return corr/len(self.phrases), lex_corr/len(self.phrases)
    

    def evaluate(self):
        if self.type == 'not-analysis':
            const_acc = self.get_const_acc()
        else:
            const_acc = 0
        em = self.get_exact_match_sentence()
        # acc, lex_acc = self.get_indiv_accuracy()
        # np_acc, nplex_acc = self.get_np_subj_accuracy()
        # vp_acc, vplex_acc = self.get_vp_verb_accuracy()
        self.results = {
            'construction accuracy': const_acc*100,
            'exact match': em*100
            # 'Tag accuracy': acc*100,
            # 'Lexical  accuracy':lex_acc*100
            # 'VP Tag Accuracy': vp_acc,
            # 'Verb Accuracy': vplex_acc
        }
        return self.results



