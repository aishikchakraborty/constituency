import os
import torch
import numpy as np
import csv
import json
import time
import logging
from tqdm import tqdm, trange
import argparse
import random
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, BartForConditionalGeneration
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from nltk.translate import bleu_score, meteor_score
from nltk.metrics.distance import edit_distance
import wandb
wandb.login()

wandb.init(project="constituency")

logging.basicConfig()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MODEL_NAMES = {
    'bart': 'facebook/bart-base',
    't5': 't5-base',
}

parser = argparse.ArgumentParser(description='Generation Project Main')
parser.add_argument('--data', type=str, default='data/', help='location of the data corpus')
parser.add_argument('--output-b', type=str, default='do', help='base transformations')
parser.add_argument('--output-nb', type=str, default='cleft', help='non-base transformations')
parser.add_argument('--model-type', type=str, default='normal', help='normal (tokens)|hw (head words)')
parser.add_argument('--plm-type', type=str, default='', help='type of model BART|T5|')
parser.add_argument('--load-checkpoint', type=str, default=None, help='load checkpoint for finetuning')
parser.add_argument('--random_seed', type=int, default=13370, help='random seed')
parser.add_argument('--numpy_seed', type=int, default=1337, help='numpy random seed')
parser.add_argument('--torch_seed', type=int, default=133, help='pytorch random seed')
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
parser.add_argument('--grad-clip', type=float, default=0.25, help='Gradient clipping')
parser.add_argument('--batch-size', type=int, default=32, help='Batch Size')
parser.add_argument('--epochs', type=int, default=5, help='Epochs')
parser.add_argument('--gpu', type=int, default=0, help='GPU id')
parser.add_argument('--max-source-length', type=int, default=100, help='Maxlen for input sentences')
parser.add_argument('--max-target-length', type=int, default=100, help='Maxlen for output sentences')
parser.add_argument('--top-k', type=int, default=5, help='Top-k sampling')
parser.add_argument('--top-p', type=float, default=0.9, help='Nucleus sampling')
parser.add_argument('--temp', type=float, default=5, help='Softmax Temperature')
parser.add_argument('--log-interval', type=int, default=40, help='Print interval')
parser.add_argument('--no-cuda', action='store_true', help='use this flag for non-gpu training')
parser.add_argument('--held-out', action='store_true', help='target transformation is held out or not')
parser.add_argument('--save', type=str, default='outputs/', help='output directory')
parser.add_argument("--do-train", action='store_true', help="Whether to run training.")
parser.add_argument("--test-seen", action='store_true', help="Whether to run on observed test set.")
parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument("--warmup-steps", default=100, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")

args = parser.parse_args()
output_file_name = f'{args.plm_type}_output_b_{args.output_b}_held_out_{args.held_out}_inputstyle_{args.model_type}.pth'
model_save_name =  args.save + output_file_name

args.plm_type = args.plm_type.lower()
# Setup CUDA, GPU & distributed training
device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
n_gpu = torch.cuda.device_count() if device == "cuda" else 0
args.device = device

logger.warning("device: %s, n_gpu: %s, 16-bits training: %s",
                device, n_gpu,  args.fp16)

model_name = MODEL_NAMES[args.plm_type]

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if args.plm_type == 't5': 
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config).to(device)
elif args.plm_type == 'bart':
    model = BartForConditionalGeneration.from_pretrained(model_name, config=config).to(device)

def set_seed(args, n_gpu):
    random.seed(args.random_seed)
    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.torch_seed)

set_seed(args, n_gpu)

if not args.held_out:
    with open(os.path.join(args.data, 'processed_train' + '.tsv'), 'r') as f:
        train_sentences = f.read().strip().split('\n')
        random.shuffle(train_sentences)
else:
    with open(os.path.join(args.data, 'processed_train_' + args.output_b + '.tsv'), 'r') as f:
        train_sentences = f.read().strip().split('\n')
if args.test_seen:
    with open(os.path.join(args.data, 'processed_test_seen_'+ args.output_nb + '_' + args.output_b + '.tsv'), 'r') as f:
        test_sentences = f.read().strip().split('\n')
else:
    with open(os.path.join(args.data, 'processed_test_'+ args.output_nb + '_' + args.output_b + '.tsv'), 'r') as f:
        test_sentences = f.read().strip().split('\n')
    
with open(os.path.join(args.data, 'processed_dev_' + args.output_b + '.tsv'), 'r') as f:
    val_sentences = f.read().strip().split('\n')


def get_sentence_lists(sentences):
    x = []
    y = []
    head_x = []
    head_y = []
    head_map_x = []
    head_map_y = []
    for i in range(len(sentences)):
        sentence_split = sentences[i].split('#')
        input_sentence = 'transformation:' + sentence_split[0] + " " + sentence_split[1]
        head_input = 'transformation:' + sentence_split[0] + " " + sentence_split[3]

        x.append(input_sentence)
        y.append(sentence_split[2])
        head_x.append(head_input)
        head_y.append(sentence_split[4])
        head_map_x.append(eval(sentence_split[5]))
        head_map_y.append(eval(sentence_split[6]))
    
    return x, y, head_x, head_y, head_map_x, head_map_y

logger.info("Loading sentences")
train_x, train_y, train_head_x, train_head_y, train_head_map_x, train_head_map_y = get_sentence_lists(train_sentences)
test_x, test_y, test_head_x, test_head_y, test_head_map_x, test_head_map_y = get_sentence_lists(test_sentences)
val_x, val_y, val_head_x, val_head_y, val_head_map_x, val_head_map_y = get_sentence_lists(val_sentences)

if args.model_type == "hw":
    train_x = train_head_x
    train_y = train_head_y
    test_x = test_head_x
    # test_y = test_head_y
    val_x = val_head_x
    val_y = val_head_y

def tokenize_lists_of_sentences(
        lists_of_sentences
        ):
    tokenized_lists_of_sentences = tokenizer(
        lists_of_sentences, 
        return_tensors="pt",
        padding="longest", 
        max_length=args.max_source_length,
        truncation=True 
        ).to(args.device)
    
    return tokenized_lists_of_sentences

def generate_batches(tokenized_source_input_ids, tokenized_source_attention_masks, tokenized_target_input_ids, batch_size):

    num_batches = len(tokenized_source_input_ids)//batch_size 
    # if tokenized_source_input_ids % batch_size == 0 else len(tokenized_source_input_ids)//batch_size + 1
    batched_tokenized_source_input_ids = [tokenized_source_input_ids[i * batch_size : (i+1) * batch_size] for i in range(num_batches)]
    batched_tokenized_source_attention_masks= [tokenized_source_attention_masks[i * batch_size : (i+1) * batch_size] for i in range(num_batches)]
    batched_tokenized_target_input_ids = [tokenized_target_input_ids[i * batch_size : (i+1) * batch_size] for i in range(num_batches)]
    return batched_tokenized_source_input_ids, batched_tokenized_source_attention_masks, batched_tokenized_target_input_ids
    
def generate_text_from_hw(sent, hw_map):
    expanded_sent = []
    expanded_phrase = ''
    # breakpoint()
    for i, token in enumerate(sent):
        if token in hw_map.keys():
            if len(expanded_phrase) > 0:
                expanded_sent.append(expanded_phrase.strip())
                expanded_phrase = ''
            expanded_sent.append(hw_map[token].strip())
        elif (expanded_phrase + token).strip() in hw_map.keys():
            expanded_sent.append(hw_map[(expanded_phrase + token).strip()])
            expanded_phrase = ''
        else:
            expanded_phrase += token + ' '
    if len(expanded_phrase) > 0:
        expanded_sent.append(expanded_phrase.strip())
    expanded_sent[-2] += expanded_sent[-1]   # dealing with punctuation correctly
    return ' '.join(expanded_sent[:-1])


class Evaluate:
    def __init__(
            self
            ):
        return

    def bleu(
            self, predictions, references
            ):
        scores = [
                bleu_score.sentence_bleu(
                    [ref.split()], pred.split(), weights = [1]
                )
                for ref, pred in zip(references, predictions)
            ]
    
        return np.mean(scores)*100.

    def meteor(
            self, predictions, references
            ):
        scores = [
                meteor_score.single_meteor_score(
                    ref.split(), pred.split()
                )
                for ref, pred in zip(references, predictions)
            ]
    
        return np.mean(scores)*100.

    def ed(
            self, predictions, references
            ):
        scores = [
                edit_distance(
                    ref, pred
                )
                for ref, pred in zip(references, predictions)
            ]
    
        return np.mean(scores)
    
eval = Evaluate()

model = model.to(args.device)

logger.info("Creating the test dataset")
tokenized_test_source = tokenize_lists_of_sentences(test_x)
tokenized_test_target = tokenize_lists_of_sentences(test_y)
tokenized_test_source_input_ids, tokenized_test_source_attention_mask = tokenized_test_source.input_ids, tokenized_test_source.attention_mask
tokenized_test_target_input_ids = tokenized_test_target.input_ids
tokenized_test_target_input_ids[tokenized_test_target_input_ids == tokenizer.pad_token_id] = -100
tokenized_test_source_input_ids, tokenized_test_source_attention_masks, tokenized_test_target_input_ids= generate_batches(tokenized_test_source_input_ids, tokenized_test_source_attention_mask, tokenized_test_target_input_ids, args.batch_size)

if args.do_train:
    logger.info("Creating the training dataset")
    tokenized_train_source = tokenize_lists_of_sentences(train_x)
    tokenized_train_target = tokenize_lists_of_sentences(train_y)
    tokenized_train_source_input_ids, tokenized_train_source_attention_mask = tokenized_train_source.input_ids, tokenized_train_source.attention_mask
    tokenized_train_target_input_ids = tokenized_train_target.input_ids
    tokenized_train_target_input_ids[tokenized_train_target_input_ids == tokenizer.pad_token_id] = -100
    tokenized_train_source_input_ids, tokenized_train_source_attention_masks, tokenized_train_target_input_ids= generate_batches(tokenized_train_source_input_ids, tokenized_train_source_attention_mask, tokenized_train_target_input_ids, args.batch_size)
    
    logger.info("Creating the dev dataset")
    tokenized_val_source = tokenize_lists_of_sentences(val_x)
    tokenized_val_target = tokenize_lists_of_sentences(val_y)
    tokenized_val_source_input_ids, tokenized_val_source_attention_mask = tokenized_val_source.input_ids, tokenized_val_source.attention_mask
    tokenized_val_target_input_ids = tokenized_val_target.input_ids
    tokenized_val_target_input_ids[tokenized_val_target_input_ids == tokenizer.pad_token_id] = -100
    tokenized_val_source_input_ids, tokenized_val_source_attention_masks, tokenized_val_target_input_ids= generate_batches(tokenized_val_source_input_ids, tokenized_val_source_attention_mask, tokenized_val_target_input_ids, args.batch_size)
     

    num_steps = (len(train_x)*args.epochs)
    dev_best_bleu = 0
    
    logger.info("Training model")
    # trainable_parameters = []
    # for param in model.model.named_parameters():
    #     assert param[1].requires_grad  # finetune all LM parameters
    #     trainable_parameters.append(param[1])
    optimizer = AdamW(model.parameters(), lr=args.lr)
        
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, num_steps)
    num_batches = len(tokenized_train_source_input_ids)
    num_dev_batches = len(tokenized_val_source_input_ids)


    
    for epoch_idx in trange(args.epochs):
        
        train_loss = []
        model.train()

        for i in tqdm(range(num_batches)):
            optimizer.zero_grad()
            output = model(input_ids=tokenized_train_source_input_ids[i], attention_mask=tokenized_train_source_attention_masks[i], labels=tokenized_train_target_input_ids[i])
            loss = output.loss
            loss = loss / args.gradient_accumulation_steps


            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            train_loss.append(loss.detach().item())
            
            if i%200==0:
                logger.info("Epoch %d: - train loss: %.4f \n" % (epoch_idx, np.average(train_loss)*args.gradient_accumulation_steps))
                
        dev_loss = []
        
        pred = []
        labels = []
        with torch.no_grad():
            model.eval()
            for j in trange(num_dev_batches):
                loss = model(input_ids=tokenized_val_source_input_ids[j], attention_mask=tokenized_val_source_attention_masks[j], labels=tokenized_val_target_input_ids[j]).loss
                output = model.generate(input_ids=tokenized_val_source_input_ids[j], attention_mask=tokenized_val_source_attention_masks[j], max_length = args.max_target_length, num_beams=1, do_sample=False, early_stopping=False)
             
                _pred = tokenizer.batch_decode(output, skip_special_tokens=True)
                tokenized_val_target_input_ids[j][tokenized_val_target_input_ids[j] == -100] = tokenizer.pad_token_id
                _labels = tokenizer.batch_decode(tokenized_val_target_input_ids[j], skip_special_tokens=True)
                dev_loss.append(loss.item())
                pred.extend(_pred)
                labels.extend(_labels)
            bleu = eval.bleu(pred, labels)
            meteor = eval.meteor(pred, labels)
            ed = eval.ed(pred, labels)

            
            if bleu > dev_best_bleu:
                dev_best_bleu = bleu
                torch.save(model.state_dict(), model_save_name)
            logger.info("Epoch %d: - dev loss: %.4f - dev bleu: %.4f - dev meteor: %.4f - dev ed: %.4f\n" % (epoch_idx, np.average(dev_loss), bleu, meteor, ed))
            dev_loss = []
            
    
test_loss = []
pred = []
labels = []
if args.load_checkpoint == None:
    del model
    if args.plm_type == 't5': 
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config).to(device)
    elif args.plm_type == 'bart':
        model = BartForConditionalGeneration.from_pretrained(model_name, config=config).to(device)
    model.load_state_dict(torch.load(model_save_name))
else:
    model.load_state_dict(torch.load(args.load_checkpoint))
model.eval()    

num_test_batches = len(tokenized_test_source_input_ids) 

predictions_data = []
test_src = []
num_batches = len(test_head_map_y)//args.batch_size 
batched_hw_map = [test_head_map_y[i * args.batch_size : (i+1) * args.batch_size] for i in range(num_batches)]
with torch.no_grad():
    for j in trange(num_test_batches):
        loss = model(input_ids=tokenized_test_source_input_ids[j], attention_mask=tokenized_test_source_attention_masks[j], labels=tokenized_test_target_input_ids[j]).loss
        output = model.generate(input_ids=tokenized_test_source_input_ids[j], attention_mask=tokenized_test_source_attention_masks[j], max_length = args.max_target_length, num_beams=1, do_sample=False, early_stopping=False)
        
        _pred = tokenizer.batch_decode(output, skip_special_tokens=True)
        if args.model_type == "hw":
            print(_pred[:5])
            print(batched_hw_map[j][:5])
            _pred = [generate_text_from_hw((_pred[k])[:-1].split() + ['.'], batched_hw_map[j][k]) for k in range(len(_pred))]
            print(_pred[:5])
            
        tokenized_test_target_input_ids[j][tokenized_test_target_input_ids[j] == -100] = tokenizer.pad_token_id
        _labels = tokenizer.batch_decode(tokenized_test_target_input_ids[j], skip_special_tokens=True)
        print(_labels[:5])
        test_loss.append(loss.item())

        src = tokenizer.batch_decode(tokenized_test_source_input_ids[j], skip_special_tokens=True)
        test_src.extend(src)
        pred.extend(_pred)
        labels.extend(_labels)
        
    bleu = eval.bleu(pred, labels)
    meteor = eval.meteor(pred, labels)
    ed = eval.ed(pred, labels)
    for i in range(len(pred)):
        predictions_data.append([test_src[i], pred[i], labels[i]])
        
    predictions_table = wandb.Table(columns=["src", "predictions", "target"], data=predictions_data)
    wandb.log({"Predictions": predictions_table})

logger.info("*"*89)      
logger.info("Test loss: %.4f - test bleu: %.4f - test meteor: %.4f - test ed: %.4f\n" % (np.average(test_loss), bleu, meteor, ed))
wandb.finish()