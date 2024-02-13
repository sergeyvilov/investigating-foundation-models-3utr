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

N_FOLDS = 10
MAX_TOK_LEN = 1024
BATCH_SIZE = 64

data_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/'

fasta = sys.argv[1]
model_dir = sys.argv[2] # data_dir + 'nucleotide-transform/nucleotide-transformer-v2-500m-multi-species'
output_dir = sys.argv[3]
fold = int(sys.argv[4])

if not '3utr' in model_dir:
    #dna model: model trained on sequences that weren't reverse complemented for genes on negative strand
    #all NT models from INstaDeep
    reverse_seq_neg_strand = True
else:
    reverse_seq_neg_strand = False

print(f'Reverse sequences on negative strand before inference: {reverse_seq_neg_strand}')

device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_dir,trust_remote_code=True).to(device);

def get_chunks(seq_tokens):
    '''
    Chunk the given token sequence into chunks of MAX_TOK_LEN
    The input sequences shouldn't contain special tokens
    The last chunk is padded with the previous chunk if it's shorter than chunk_size
    '''
    if tokenizer.eos_token_id:
        chunk_len = MAX_TOK_LEN-2 #2 special tokens to be added 
    else:
        chunk_len = MAX_TOK_LEN-1 #only cls token
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

def mask_sequence(seq_tokens, left_shift=0):
    '''
    Consecutively mask tokens in the sequence and yield each masked position
    '''    
    for mask_pos in range(1+left_shift,len(seq_tokens)):
        if seq_tokens[mask_pos] in (tokenizer.eos_token_id,tokenizer.pad_token_id):
            break
        masked_seq = seq_tokens.clone()
        masked_seq[mask_pos] = tokenizer.mask_token_id
        yield mask_pos, masked_seq

class SeqDataset(IterableDataset):
    
    def __init__(self, seq_df):
        
        self.seq_df = seq_df
        self.start = 0
        self.end = len(self.seq_df)
        
    def __iter__(self):
        
        for seq_idx in range(self.start, self.end):
            
            seq_info = self.seq_df.iloc[seq_idx]
            chunk, left_shift = seq_info.seq
            
            #gt_tokens = chunk + [tokenizer.pad_token_id]*(MAX_TOK_LEN-len(chunk))
            gt_tokens = torch.LongTensor(chunk)
            
            for masked_pos, masked_tokens in mask_sequence(gt_tokens, left_shift):
                assert masked_tokens[masked_pos] == tokenizer.mask_token_id
                yield seq_info.seq_name, gt_tokens, masked_pos, masked_tokens

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
    compl_dict = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}
    compl_seq = ''.join([compl_dict.get(x,x) for x in seq])
    rev_seq = compl_seq[::-1]
    return rev_seq

tokendict_list = [{"A": [], "G": [], "T": [],"C": []} for x in range(6)]

for tpl in itertools.product("ACGT",repeat=6):
    encoding = tokenizer.encode("".join(tpl))
    for idx, nuc in enumerate(tpl):
        tokendict_list[idx][nuc].append(encoding[1]) #token indices for idx position in 6-mer and letter nuc

seq_df = defaultdict(str)

with open(fasta, 'r') as f:
    for line in f:
        if line.startswith('>'):
            seq_name = line[1:].rstrip()
        else:
            seq_df[seq_name] += line.rstrip().upper()
            
seq_df = pd.DataFrame(list(seq_df.items()), columns=['seq_name','seq'])

folds = np.arange(N_FOLDS).repeat(len(seq_df)//N_FOLDS+1)[:len(seq_df)] 

seq_df = seq_df.loc[folds==fold]

print(f'Fold {fold}: {len(seq_df)} sequences')

strand_info = pd.read_csv(data_dir + 'UTR_coords/GRCh38_3_prime_UTR_clean-sorted.bed', sep='\t', header = None, names=['seq_name','strand'], usecols=[3,5]).set_index('seq_name').squeeze()

seq_df.seq = seq_df.apply(lambda x: reverse_complement(x.seq) if strand_info.loc[x.seq_name]=='-' and reverse_seq_neg_strand else x.seq, axis=1) #undo reverse complement

original_seqs = seq_df.set_index('seq_name').seq

seq_df['seq'] = seq_df.seq.apply(lambda x: tokenizer(x,add_special_tokens=False)['input_ids'])
seq_df['seq'] = seq_df.seq.apply(lambda x:get_chunks(x))

seq_df = seq_df.explode('seq')

def collate_fn(batch):
    seq_names_batch, gt_tokens_batch, masked_pos_batch, masked_tokens_batch = zip(*batch)
    masked_tokens_batch = pad_sequence(masked_tokens_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    gt_tokens_batch = pad_sequence(gt_tokens_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    return seq_names_batch, gt_tokens_batch, masked_pos_batch, masked_tokens_batch

dataloader = DataLoader(SeqDataset(seq_df), batch_size=BATCH_SIZE,shuffle=False, collate_fn=collate_fn, num_workers=1, worker_init_fn=worker_init_fn)

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
        gt_token = tokenizer.id_to_token(gt_tokens[masked_pos].item())
        for idx in range(len(gt_token)):
            position_probas = [] #probabilities for all bases at given position
            for nuc in 'ACGT':
                position_probas.append(seq_probas[masked_pos][tokendict_list[idx][nuc]].sum()) #sum over all takens that have given letter at given position
            all_probas[seq_name].append(position_probas)
        if seq_name!=prev_seq_name:
            if len(verif_seqs[prev_seq_name])>0:
                is_correct.extend([nuc_dict.get(base,4)==gt_idx for base, gt_idx in zip(verif_seqs[prev_seq_name],np.argmax(all_probas[prev_seq_name],axis=1))])
                print(f'Sequence {prev_seq_name} processed ({len(verif_seqs)-1}/{len(original_seqs)}), loss: {np.mean(all_losses):.3}, acc:{np.mean(is_correct):.3}')
                assert verif_seqs[prev_seq_name]==original_seqs.loc[prev_seq_name]
                #pbar.update(1)
            prev_seq_name = seq_name
        verif_seqs[seq_name] += gt_token      

assert verif_seqs[seq_name]==original_seqs.loc[seq_name]

seq_names = list(all_probas.keys())
probs = [np.array(x) for x in all_probas.values()]
seqs = original_seqs.loc[seq_names].values.tolist()

if reverse_seq_neg_strand:
    probs = [x[::-1,[3,2,1,0]] if strand_info.loc[seq_name]=='-' else x for x, seq_name in zip(probs,seq_names)]
    seqs = [reverse_complement(x) if strand_info.loc[seq_name]=='-' else x for x, seq_name in zip(seqs,seq_names)]

os.makedirs(output_dir, exist_ok=True)

with open(output_dir + f'/predictions_{fold}.pickle', 'wb') as f:
    pickle.dump({'seq_names':seq_names, 'seqs':seqs, 'probs':probs, 'fasta':fasta},f)

print('Done')




