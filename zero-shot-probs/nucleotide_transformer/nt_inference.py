#!/usr/bin/env python
# coding: utf-8

import torch 
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence

import itertools

import os, sys

import argparse

def str_to_bool(value):
    return value.lower() in {'true', 't', '1', 'yes', 'y'}

parser = argparse.ArgumentParser("main.py")

parser.add_argument("--fasta", help = "fasta file name", type = str, required = True)

parser.add_argument("--checkpoint_dir", help = "checkpoint dir", type = str, required = True)

parser.add_argument("--output_dir", help = "output dir name", type = str, required = True)

parser.add_argument("--batch_size", help = "batch size", type = int, default = 64, required = False)

parser.add_argument("--max_tok_len", help = "maximum chunk length", type = int, default = 1024, required = False)

parser.add_argument("--N_folds", help = "number of folds to split sequences", type = int, default = None, required = False)

parser.add_argument("--fold", help = "current fold", type = int, default = None, required = False)

parser.add_argument("--masking", help = "consecutively mask tokens for inference", type = str_to_bool, default = False, required = False)

parser.add_argument("--ref_aware", help = "use reference-aware decoding", type = str_to_bool, default = False, required = False)

parser.add_argument("--reverse_seq_neg_strand", help = "reverse-complement sequences on the negative strand before inference", action='store_true', default = False, required = False)

parser.add_argument("--strand_bed", help = "bed file with sequence strand information, used with reverse_seq_neg_strand ", type = str, required = False)

parser.add_argument("--whitelist", help = "include sequences only from this list", type = str, default=None, required = False)

parser.add_argument("--blacklist", help = "exclude sequences from this list", type = str, default=None, required = False)

parser.add_argument("--central_window", help = "perform inference only for central_window nucleotides around the sequence center, assign nan probability to all other positions", type = int, default=False, required = False)

parser.add_argument("--predict_only_lowercase", help = "keep predictions only for the lowercased positions", action='store_true', default = False, required = False)

input_params = parser.parse_args()

assert not (input_params.predict_only_lowercase and input_params.central_window)

print(input_params)

print(f'Reverse sequences on negative strand before inference: {input_params.reverse_seq_neg_strand}')

device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(input_params.checkpoint_dir,trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(input_params.checkpoint_dir,trust_remote_code=True).to(device);

def get_chunks(seq_tokens):
    '''
    Chunk the given token sequence into chunks of MAX_TOK_LEN
    The input sequences shouldn't contain special tokens
    The last chunk is padded with the previous chunk if it's shorter than chunk_size
    '''
    if tokenizer.eos_token_id:
        chunk_len = input_params.max_tok_len-2 #2 special tokens to be added 
    else:
        chunk_len = input_params.max_tok_len-1 #only cls token
    chunks = [seq_tokens[start:start+chunk_len] for start in range(0,len(seq_tokens),chunk_len)]
    assert [x for y in chunks for x in y]==seq_tokens
    if len(chunks)>1:
        left_shift = min(chunk_len-len(chunks[-1]), len(chunks[-2]))
        if left_shift>0:
            pad_seq = chunks[-2][-left_shift:]
            chunks[-1] = pad_seq + chunks[-1]
    else:
        left_shift = 0
    if tokenizer.eos_token_id:
        chunks = [[tokenizer.cls_token_id, *chunk, tokenizer.eos_token_id] for chunk in chunks]
        assert [x for y in chunks[:-1] for x in y[1:-1]]+[x for x in  chunks[-1][1+left_shift:-1]]==seq_tokens
    else:
        chunks = [[tokenizer.cls_token_id, *chunk] for chunk in chunks]
        assert [x for y in chunks[:-1] for x in y[1:]]+[x for x in  chunks[-1][1+left_shift:]]==seq_tokens
    return [(chunk,0) if chunk_idx!=len(chunks)-1 else (chunk,left_shift) for chunk_idx, chunk in enumerate(chunks)]

def mask_sequence(seq_tokens, mask_crop_left=0,mask_crop_right=None):
    '''
    Consecutively mask tokens in the sequence and yield each masked position
    Mask tokens between mask_crop_left and mask_crop_right
    Don't mask special tokens
    '''    
    if not mask_crop_right:
        mask_crop_right = len(seq_tokens)-1
    for mask_pos in range(1+mask_crop_left,1+mask_crop_right):
        if seq_tokens[mask_pos] in (tokenizer.eos_token_id,tokenizer.pad_token_id):
            break
        masked_seq = seq_tokens.clone()
        masked_seq[mask_pos] = tokenizer.mask_token_id
        yield mask_pos, masked_seq

class SeqDataset(IterableDataset):
    
    def __init__(self, seq_df, masking=True):
        
        self.seq_df = seq_df
        self.start = 0
        self.end = len(self.seq_df)
        self.masking = masking
        
    def __iter__(self):
        
        for seq_idx in range(self.start, self.end):
            
            seq_info = self.seq_df.iloc[seq_idx]
            chunk = seq_info.tokens
            
            gt_tokens = torch.LongTensor(chunk)

            mask_crop_left = seq_info.crop_mask_left
            mask_crop_right = seq_info.crop_mask_right
                
            if self.masking:
                for masked_pos, masked_tokens in mask_sequence(gt_tokens, mask_crop_left, mask_crop_right):
                    #consecutively mask each token in the sequence
                    assert masked_tokens[masked_pos] == tokenizer.mask_token_id
                    yield seq_info.name, gt_tokens, masked_pos, masked_tokens
            else:
                yield seq_info.name, gt_tokens, -1, gt_tokens

def worker_init_fn(worker_id):
     worker_info = torch.utils.data.get_worker_info()
     dataset = worker_info.dataset  # the dataset copy in this worker process
     overall_start = dataset.start
     overall_end = dataset.end
     # configure the dataset to only process the split workload
     per_worker = int(np.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
     worker_id = worker_info.id
     dataset.start = overall_start + worker_id * per_worker
     dataset.end = min(dataset.start + per_worker, overall_end)

def worker_init_fn(worker_id):
     worker_info = torch.utils.data.get_worker_info()
     dataset = worker_info.dataset  # the dataset copy in this worker process
     overall_start = dataset.start
     overall_end = dataset.end
     # configure the dataset to only process the split workload
     per_worker = int(np.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
     worker_id = worker_info.id
     dataset.start = overall_start + worker_id * per_worker
     dataset.end = min(dataset.start + per_worker, overall_end)

def crop_seq(seq,tokens):

    L = len(seq)
    
    if input_params.central_window:
        left = L//2-input_params.central_window//2
        right = left+input_params.central_window
    elif input_params.predict_only_lowercase:
        lower_idx = np.array([idx for idx, c in enumerate(seq) if c.islower()])
        left = lower_idx.min()
        right = lower_idx.max()

    decoded_tokens = tokenizer.decode(tokens).split()
    
    nt_idx = []
    for token_idx,token in enumerate(decoded_tokens[1:]):
        if not token.startswith('<'):
            nt_idx.extend([token_idx]*len(token))
            
    nt_idx = np.array(nt_idx)
    
    crop_mask_left, crop_mask_right = nt_idx[left],nt_idx[right]+1
    seq_idx = np.where((nt_idx>=crop_mask_left) & (nt_idx<crop_mask_right))[0]
    
    pos_left,pos_right = seq_idx[0], seq_idx[-1]+1
    
    seq_cropped = seq[pos_left:pos_right]
    
    assert seq_cropped.startswith(seq[pos_left:pos_left+right-left])
    
    return (seq, tokens, L, seq_cropped,pos_left, pos_right, crop_mask_left, crop_mask_right)
    
def predict_on_batch(masked_tokens_batch):

    targets_masked = masked_tokens_batch.clone()
    targets_masked[targets_masked!=tokenizer.mask_token_id] = -100
    attention_mask = masked_tokens_batch!= tokenizer.pad_token_id   
    
    with torch.no_grad():
        torch_outs = model(
        masked_tokens_batch.to(device),
        labels = targets_masked.to(device),
        attention_mask=attention_mask.to(device),
        encoder_attention_mask=attention_mask.to(device),
        output_hidden_states=False)
    
    logits = torch_outs["logits"] #max_tokenized_length x (max_tokenized_length+1) x N_tokens
    
    probas_batch = F.softmax(logits, dim=-1).cpu().numpy()
    
    loss = torch_outs["loss"].item()
    
    return probas_batch, loss

def reverse_complement(seq):
    '''
    Take sequence reverse complement
    '''
    compl_dict = {'A':'T', 'C':'G', 'G':'C', 'T':'A','a':'t', 'c':'g', 'g':'c', 't':'a'}
    compl_seq = ''.join([compl_dict.get(x,x) for x in seq])
    rev_seq = compl_seq[::-1]
    return rev_seq

tokendict_list = [{"A": [], "G": [], "T": [],"C": []} for x in range(6)]

for tpl in itertools.product("ACGT",repeat=6):
    encoding = tokenizer.encode("".join(tpl))
    for idx, nuc in enumerate(tpl):
        tokendict_list[idx][nuc].append(encoding[1]) #token indices for idx position in 6-mer and letter nuc

seq_df = defaultdict(str)

with open(input_params.fasta, 'r') as f:
    for line in f:
        if line.startswith('>'):
            seq_name = line[1:].rstrip()
        else:
            seq_df[seq_name] += line.rstrip()#.upper()
            
seq_df = pd.DataFrame(list(seq_df.items()), columns=['seq_name','seq']).set_index('seq_name')

if input_params.whitelist:
    whitelist = pd.read_csv(input_params.whitelist,header=None,names=['seq_name']).seq_name.values
    seq_df = seq_df[seq_df.index.isin(whitelist)]
    print(len(seq_df))

if input_params.blacklist:
    blacklist = pd.read_csv(input_params.blacklist,header=None,names=['seq_name']).seq_name.values
    seq_df = seq_df[~seq_df.index.isin(blacklist)]
    print(len(seq_df))

if input_params.N_folds:
    folds = np.arange(input_params.N_folds).repeat(len(seq_df)//input_params.N_folds+1)[:len(seq_df)] 
    
    seq_df = seq_df.loc[folds==input_params.fold]
    
    print(f'Fold {input_params.fold}: {len(seq_df)} sequences')

if input_params.reverse_seq_neg_strand:
    strand_info = pd.read_csv(input_params.strand_bed, sep='\t', header = None, names=['seq_name','strand'], usecols=[3,5]).set_index('seq_name').squeeze()
    
    seq_df.seq = seq_df.apply(lambda x: reverse_complement(x.seq) if strand_info.loc[x.name]=='-' else x.seq, axis=1) #undo reverse complement

original_seqs = seq_df.seq #sequences before tokenization

if input_params.central_window or input_params.predict_only_lowercase:
    seq_df['tokens'] = seq_df.seq.apply(lambda seq:tokenizer(seq.upper(),add_special_tokens=True)['input_ids'])
    seq_df = pd.DataFrame([crop_seq(seq,tokens) for seq,tokens in seq_df.values], index=seq_df.index, columns=['seq','tokens','seq_length','seq_cropped','pos_left','pos_right','crop_mask_left','crop_mask_right'])
else:
    tokens = [(seq_name,chunk[0],chunk[1]) for seq_name,seq in seq_df.seq.items() for chunk in get_chunks(tokenizer(seq.upper(),add_special_tokens=False)['input_ids'])
             ]
    seq_df = pd.DataFrame(tokens,columns=['seq_name','tokens','crop_mask_left']).set_index('seq_name')
    seq_df['crop_mask_right'] = seq_df.tokens.apply(len)-1
 
def collate_fn(batch):
    seq_names_batch, gt_tokens_batch, masked_pos_batch, masked_tokens_batch = zip(*batch)
    masked_tokens_batch = pad_sequence(masked_tokens_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    gt_tokens_batch = pad_sequence(gt_tokens_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    return seq_names_batch, gt_tokens_batch, masked_pos_batch, masked_tokens_batch

def predict_probas_token(seq_token_probas,token_pos,gt_token):
    '''
    Predict probabilities of each bp for a given token
    '''
    seq_probas = []
    for idx in range(len(gt_token)):
        #loop over all positions of the masked token
        position_probas = [] #probabilities for all bases at given position
        for nuc in 'ACGT':
            if input_params.ref_aware:
                token_idx = [tokenizer.token_to_id(gt_token[:idx]+nuc+gt_token[idx+1:])] #single token 
            else:
                token_idx = tokendict_list[idx][nuc] #all tokens that have given base nuc at given position idx
            position_probas.append(seq_token_probas[token_pos][token_idx].sum()) 
        seq_probas.append(position_probas)
    return seq_probas
    
dataloader = DataLoader(SeqDataset(seq_df,masking=input_params.masking), batch_size=input_params.batch_size,shuffle=False, collate_fn=collate_fn, num_workers=1, worker_init_fn=worker_init_fn)

nuc_dict = {"A":0,"C":1,"G":2,"T":3}

all_probas = defaultdict(list)
verif_seqs = defaultdict(str)

all_losses, is_correct = [], []
prev_seq_name = None

n_ready = 0

#pbar = tqdm(total=len(original_seqs))

for seq_names_batch, gt_tokens_batch, masked_pos_batch, masked_tokens_batch in dataloader:

    probas_batch, loss_batch = predict_on_batch(masked_tokens_batch)
    #probas_batch, loss_batch = np.zeros((len(seq_names_batch),1024,4108)), 0
    
    all_losses.append(loss_batch)
    
    for seq_name, gt_tokens, masked_pos, seq_probas in zip(seq_names_batch, gt_tokens_batch, masked_pos_batch, probas_batch):
        gt_tokens = gt_tokens.cpu().tolist()
        if input_params.masking:
            gt_token = tokenizer.id_to_token(gt_tokens[masked_pos]) #ground truth masked token
            all_probas[seq_name].extend(predict_probas_token(seq_probas,masked_pos,gt_token))
            verif_seqs[seq_name] += gt_token
        else:
            for token_idx, gt_token in enumerate(gt_tokens):
                gt_token = tokenizer.id_to_token(gt_token) #ground truth token
                if not gt_token.startswith('<'):
                    all_probas[seq_name].extend(predict_probas_token(seq_probas,token_idx,gt_token))
                    verif_seqs[seq_name] += gt_token
        if seq_name!=prev_seq_name:
            if len(verif_seqs[prev_seq_name])>0:
                is_correct.extend([nuc_dict.get(base,4)==gt_idx for base, gt_idx in zip(verif_seqs[prev_seq_name],np.argmax(all_probas[prev_seq_name],axis=1))])
                print(f'Sequence {prev_seq_name} processed ({len(verif_seqs)-1}/{len(original_seqs)}), loss: {np.mean(all_losses):.3}, acc:{np.mean(is_correct):.3}')
                if input_params.central_window or input_params.predict_only_lowercase:
                    assert verif_seqs[prev_seq_name]==seq_df.loc[prev_seq_name]['seq_cropped'].upper() #compare reconstruction from the masked token with the original sequence
                else:
                    assert verif_seqs[prev_seq_name]==original_seqs.loc[prev_seq_name].upper() #compare reconstruction from the masked token with the original sequence
                #pbar.update(1)
            prev_seq_name = seq_name

seq_names = list(all_probas.keys())
probs = [np.array(x) for x in all_probas.values()]
seqs = original_seqs.loc[seq_names].values.tolist()

if input_params.central_window or input_params.predict_only_lowercase:
    for seq_idx,seq_name in enumerate(seq_names):
        pad_left = np.empty((seq_df.loc[seq_name]['pos_left'],4))
        pad_right = np.empty((seq_df.loc[seq_name]['seq_length']-seq_df.loc[seq_name]['pos_right'],4))
        pad_left[:] = np.nan
        pad_right[:] = np.nan
        probs[seq_idx] = np.vstack((pad_left,probs[seq_idx],pad_right))
        
if input_params.reverse_seq_neg_strand:
    probs = [x[::-1,[3,2,1,0]] if strand_info.loc[seq_name]=='-' else x for x, seq_name in zip(probs,seq_names)]
    seqs = [reverse_complement(x) if strand_info.loc[seq_name]=='-' else x for x, seq_name in zip(seqs,seq_names)]

os.makedirs(input_params.output_dir, exist_ok=True)

if input_params.N_folds:
    output_name = input_params.output_dir + f'/predictions_{input_params.fold}.pickle'
else:
    output_name = input_params.output_dir + f'/predictions.pickle'
with open(output_name, 'wb') as f:
    pickle.dump({'seq_names':seq_names, 'seqs':seqs, 'probs':probs, 'fasta':input_params.fasta},f)

print('Done')




