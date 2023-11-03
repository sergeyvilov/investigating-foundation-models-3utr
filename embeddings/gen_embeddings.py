#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import defaultdict

import os,sys
import builtins
import time
import pickle

import torch 
from transformers import AutoTokenizer, AutoModelForMaskedLM,AutoModel
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

sys.path.append('DNABERT/')

from src.transformers import DNATokenizer 
from transformers import BertModel, BertConfig


class dotdict(dict):
    '''
    dot.notation access to dictionary attributes
    '''
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# In[2]:


data_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/MLM/'


# In[3]:

import argparse

parser = argparse.ArgumentParser("main.py")

parser.add_argument("--fasta", help = "fasta file name", type = str, required = True)

parser.add_argument("--model", help = "model name: NT-MS-v2-500M, DNABERT, DNABERT-2", type = str, required = True)

parser.add_argument("--output_dir", help = "output dir name", type = str, required = True)

parser.add_argument("--batch_size", help = "batch size", type = int, default = 10, required = False)

parser.add_argument("--N_folds", help = "number of folds to split sequences", type = int, default = None, required = False)

parser.add_argument("--fold", help = "current fold", type = int, default = None, required = False)

parser.add_argument("--exclude", help = "list of sequences to exclude", type = str, default = None, required = False)

input_params = vars(parser.parse_args())

input_params = dotdict(input_params)

for key,value in input_params.items():
    print(f'{key}: {value}')
    

# In[4]:
MAX_SEQ_LENGTH = 5000



# In[5]:


def load_model(model_name):
    
    model_dirs = {'DNABERT':data_dir + 'dnabert/default/6-new-12w-0/',
                  'DNABERT-2':data_dir + 'dnabert2/DNABERT-2-117M/',
                  'NT-MS-v2-500M':data_dir + 'nucleotide-transform/nucleotide-transformer-v2-500m-multi-species'} 

    if model_name == 'DNABERT':
        
        config = BertConfig.from_pretrained('https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/config.json')
        tokenizer = DNATokenizer.from_pretrained('dna6')
        model = BertModel.from_pretrained(model_dirs[model_name], config=config)

    elif model_name == 'DNABERT-2':

        tokenizer = AutoTokenizer.from_pretrained(model_dirs[model_name],trust_remote_code=True)
        model = AutoModel.from_pretrained(model_dirs[model_name],trust_remote_code=True)

    elif model_name == 'NT-MS-v2-500M':

        # Import the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(model_dirs[model_name],trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_dirs[model_name],trust_remote_code=True)

    return tokenizer, model


# In[6]:


class SeqDataset(Dataset):
    
    def __init__(self, fasta_file):
        
        seqs = defaultdict(str)
            
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    transcript_id = line[1:].split(':')[0].rstrip()
                else:
                    seqs[transcript_id] += line.rstrip().upper()
                    
        seqs = {k:v[:MAX_SEQ_LENGTH] for k,v in seqs.items()}
        #seqs = {k:''.join(np.random.choice(list('ACGT'),size=MAX_LENGTH)) for k,v in seqs.items()} 
        
        seqs = list(seqs.items())
        
        if input_params.exclude!=None:
            print(f'Excluding sequences from {input_params.processed_seqs}')
            processed_seqs = pd.read_csv(input_params.exclude,names=['seq_name']).seq_name.values
            seqs = [(seq_name,seq) for seq_name,seq in seqs if not seq_name in processed_seqs]
        if input_params.N_folds!=None:
            print(f'Fold {input_params.fold}')
            folds = np.repeat(np.arange(input_params.N_folds),len(seqs)//input_params.N_folds+1)[:len(seqs)]
            seqs = [x for idx,x in enumerate(seqs) if folds[idx]==input_params.fold]
            
        self.seqs = seqs
        self.max_length = max([len(seq[1]) for seq in self.seqs])
        
    def __len__(self):
        
        return len(self.seqs)
    
    def __getitem__(self, idx):
        
        return self.seqs[idx]

# In[7]:


def kmers_stride1(seq, k=6):
    # splits a sequence into overlapping k-mers
    return [seq[i:i + k] for i in range(0, len(seq)-k+1)] 


def get_batch_embeddings(model_name, sequences):

    if model_name == 'DNABERT':

        outputs = []
 
        for seq in sequences:

            seq_kmer = kmers_stride1(seq)
    
            model_input = tokenizer.encode_plus(seq_kmer, add_special_tokens=True, padding='max_length', max_length=512)["input_ids"]
            model_input = torch.tensor(model_input, dtype=torch.long)
            model_input = model_input.unsqueeze(0)   # to generate a fake batch with batch size one

            output = model(model_input)
            outputs.append(output[1])

        return torch.vstack(outputs)

    elif model_name == 'DNABERT-2':

        inputs = tokenizer(sequences, return_tensors = 'pt', padding="max_length", max_length = dataset.max_length)["input_ids"]
        
        hidden_states = model(inputs)[0] # [1, sequence_length, 768]
        
        # embedding with mean pooling
        mean_sequence_embeddings = torch.mean(hidden_states, dim=1)

        return mean_sequence_embeddings

    elif model_name == 'NT-MS-v2-500M':

        batch_token_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="max_length", max_length = dataset.max_length)["input_ids"]

        attention_mask = batch_token_ids != tokenizer.pad_token_id
            
        torch_outs = model(
            batch_token_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True)
        
        # Compute sequences embeddings
        embeddings = torch_outs['hidden_states'][-1].detach().numpy()
        #print(f"Embeddings shape: {embeddings.shape}")
        #print(f"Embeddings per token: {embeddings}")
        
        # Add embed dimension axis
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)
        
        # Compute mean embeddings per sequence
        mean_sequence_embeddings = torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)
        #print(f"Mean sequence embeddings: {mean_sequence_embeddings}")

        probas = F.softmax(torch_outs['logits'],dim=2).cpu().numpy()
        batch_token_ids = batch_token_ids.cpu().numpy()
        gt_probas = np.take_along_axis(probas, batch_token_ids[...,None], axis=2).squeeze()
        log_probas = np.log(gt_probas)

    return (mean_sequence_embeddings, log_probas)


# In[8]:


#sequences = next(iter(dataloader))


# In[9]:


tokenizer, model = load_model(input_params.model)


# In[10]:


dataset = SeqDataset(input_params.fasta)

dataloader = DataLoader(dataset = dataset, 
                        batch_size = input_params.batch_size, 
                        num_workers = 2, collate_fn = None, shuffle = False)


# In[11]:
print(f'Total sequences: {len(dataset)}')

os.makedirs(input_params.output_dir, exist_ok=True)

all_emb = []
all_logprobs = []

def print(*args, **kwargs):
    '''
    Redefine print function for logging
    '''
    now = time.strftime("[%Y/%m/%d-%H:%M:%S]-", time.localtime()) #current date and time at the beggining of each printed line
    builtins.print(now, *args, **kwargs)
    sys.stdout.flush()



for seq_idx, (seq_names,sequences) in enumerate(dataloader):

    if os.path.isfile(input_params.output_dir + f'/{seq_names[0]}.pickle'):
        continue
        
    print(f'generating embeddings for batch {seq_idx}/{len(dataloader)}')
    
    with torch.no_grad():
        outputs = get_batch_embeddings(input_params.model,sequences)

    if isinstance(outputs, tuple):
       emb, logprobs = outputs
       all_logprobs.append(logprobs)
    else:
       emb, logprobs = outputs, []
    
    emb = emb.cpu().numpy()

    all_emb.append(emb)

    #with open(input_params.output_dir + f'/{seq_names[0]}.pickle', 'wb') as f:
    #    pickle.dump((seq_names,emb,logprobs),f)

if input_params.fold!=None:
    output_name = input_params.output_dir + f'/embeddings_{input_params.fold}.npy'
else:
    output_name = input_params.output_dir + '/embeddings.npy'

with open(output_name, 'wb') as f:
    np.save(f, np.vstack(all_emb))


