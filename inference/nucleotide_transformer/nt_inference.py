#!/usr/bin/env python
# coding: utf-8

# In[19]:


import torch 
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForMaskedLM

import numpy as np
import pandas as pd
import pickle

import itertools

import os, sys



# In[44]:


chunk_size = 1000
N_folds = 10


# In[3]:




# In[4]:


# In[ ]:

dataset = sys.argv[1]
model_dir = sys.argv[2] # data_dir + 'nucleotide-transform/nucleotide-transformer-v2-500m-multi-species'
output_dir = sys.argv[3]
fold = int(sys.argv[4])

data_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/MLM/'


# In[5]:


# Import the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_dir,trust_remote_code=True)


# In[6]:


def generate_batch(seq, max_seq_length):
    '''
    Generate a batch by consequently masking every 6-mer in a sequence in a rolling mask fashion
    '''

    max_tokenized_length = max_seq_length-(max_seq_length//6)*5 #maximum length of the tokenized sequence without 'N's
    
    seq_token_ids = tokenizer.encode_plus(seq, return_tensors="pt", padding="max_length", max_length = max_tokenized_length)["input_ids"].squeeze()
    
    #mask_id = tokenizer.token_to_id('<mask>')
    #pad_id = tokenizer.token_to_id('<pad>')
    
    batch_token_masked = []
    
    for mask_pos in range(1,len(seq_token_ids)):
        if seq_token_ids[mask_pos] == tokenizer.pad_token_id:
            break
        masked_seq = seq_token_ids.clone()
        masked_seq[mask_pos] = tokenizer.mask_token_id
        batch_token_masked.append(masked_seq)
    
    batch_token_masked = torch.stack(batch_token_masked)

    seq_token_ids = seq_token_ids.numpy()

    return seq_token_ids, batch_token_masked #unmasked tokens for the sequence, batch of masked positions


# In[7]:


def predict_on_batch(seq_token_ids, batch_token_ids):
    '''
    Predict on a batch corresponding to a single sequence
    '''

    with torch.no_grad():
        attention_mask = batch_token_ids != tokenizer.pad_token_id   
        torch_outs = model(
        batch_token_ids,
        attention_mask=attention_mask,
        encoder_attention_mask=attention_mask,
        output_hidden_states=False)

    logits = torch_outs["logits"] #max_tokenized_length x (max_tokenized_length+1) x N_tokens

    probas = F.softmax(logits, dim=-1).numpy()

    seq_probas = []
    
    for masked_pos, gt_token_id in enumerate(seq_token_ids[1:]): #loop over tokens of unmasked sequence
        gt_token = tokenizer.id_to_token(gt_token_id)
        if gt_token=='<pad>':
            break
        assert batch_token_masked[masked_pos,masked_pos+1]==2 #masked position
        for idx in range(len(gt_token)):
            position_probas = [] #probabilities for all bases at given position
            for nuc in 'ACGT':
                position_probas.append(probas[masked_pos,masked_pos+1][tokendict_list[idx][nuc]].sum()) #sum over all takens that have given letter at given position
            seq_probas.append(position_probas)
    
    seq_probas = np.array(seq_probas)

    return seq_probas


# In[8]:


def reverse_complement(seq):
    '''
    Take sequence reverse complement
    '''
    compl_dict = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}
    compl_seq = ''.join([compl_dict.get(x,x) for x in seq])
    rev_seq = compl_seq[::-1]
    return rev_seq


# In[9]:


tokendict_list = [{"A": [], "G": [], "T": [],"C": []} for x in range(6)]

for tpl in itertools.product("ACGT",repeat=6):
    encoding = tokenizer.encode("".join(tpl))
    for idx, nuc in enumerate(tpl):
        tokendict_list[idx][nuc].append(encoding[1]) #token indices for idx position in 6-mer and letter nuc


# In[10]:


def get_chunks(seq,chunk_size):
    '''
    Chunk the given sequence into chunks of chunk_size
    The last chunk is padded with the previous chunk if it's shorter than chunk_size
    '''
    chunks = [seq[start:start+chunk_size] for start in range(0,len(seq),chunk_size)]
    assert ''.join(chunks)==seq
    if len(chunks)>1:
        pad_length_last = min(chunk_size-len(chunks[-1]), len(chunks[-2]))
        if pad_length_last>0:
            pad_seq = chunks[-2][-pad_length_last:]
            chunks[-1] = pad_seq + chunks[-1]
    else:
        pad_length_last = 0
    assert ''.join([x for x in chunks[:-1]]+[chunks[-1][pad_length_last:]])==seq
    return (chunks,pad_length_last)


# In[51]:


dataset = pd.read_csv(dataset)

folds = np.arange(N_folds).repeat(len(dataset)//N_folds+1)[:len(dataset)] 

dataset = dataset.loc[folds==fold]

print(f'Fold {fold}: {len(dataset)} sequences')

strand_info = pd.read_csv(data_dir + 'UTR_coords/GRCh38_3_prime_UTR_clean-sorted.bed', sep='\t', header = None, names=['seq_name','strand'], usecols=[3,5]).set_index('seq_name').squeeze()

dataset.seq = dataset.apply(lambda x: reverse_complement(x.seq) if strand_info.loc[x.seq_name]=='-' else x.seq, axis=1) #undo reverse complement

dataset = dataset.set_index('seq_name').seq

seq_length = dataset.apply(len)

dataset = dataset.apply(lambda x:get_chunks(x,chunk_size))



# In[52]:


#(seq_chunks,pad_length_last) = dataset.iloc[0]


# In[53]:


all_preds = []


for seq_name, (seq_chunks,pad_length_last) in dataset.items():
    if not os.path.isfile(output_dir + f'/{seq_name}.pickle'):
        print(f'Predicting for {seq_name}, {len(seq_chunks)} chunks')
        seq_probas = []
        for seq in seq_chunks:
            seq_token_ids, batch_token_masked = generate_batch(seq,chunk_size)
            seq_probas.append(predict_on_batch(seq_token_ids, batch_token_masked))
        seq_probas[-1] = seq_probas[-1][pad_length_last:] #skip the part used for padding from the previous chunk
        seq_probas = np.vstack(seq_probas)
        assert sum([len(x) for x in seq_chunks])-pad_length_last==seq_probas.shape[0]
        assert seq_length[seq_name] == seq_probas.shape[0]
        if strand_info[seq_name]=='-':
            seq_probas = seq_probas[::-1,[3,2,1,0]] #reverse complement probabilities
        all_preds.append((seq_name,seq_probas))
        with open(output_dir + f'/{seq_name}.pickle', 'wb') as f:
            pickle.dump((seq_name,seq_probas),f)


# In[54]:


with open(output_dir + f'/fold_{fold}.pickle', 'wb') as f:
    pickle.dump(all_preds,f)

print('Done')


# In[ ]:




