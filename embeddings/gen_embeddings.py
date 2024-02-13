#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings
import pickle

from collections import defaultdict

import os

import torch 
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, BertForMaskedLM

#from DNABERT2.bert_layers import BertModel as DNABERT2
from DNABERT.src.transformers.tokenization_dna import DNATokenizer

import helpers.misc as misc


data_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/mlm/'

model_dirs = {
'dnabert':data_dir + 'models/dnabert/6-new-12w-0/',
'dnabert2':data_dir + 'models/dnabert2/DNABERT-2-117M/',
'ntrans-v2-250m':data_dir + 'models/nucleotide-transformer-v2-250m-multi-species',
'ntrans-v2-500m':data_dir + 'models/nucleotide-transformer-v2-500m-multi-species',
'dnabert-3utr':data_dir + 'models/dnabert-3utr/checkpoints/epoch_30/',
'dnabert2-3utr':data_dir + 'models/dnabert2-3utr/checkpoints/epoch_18/',
'ntrans-v2-250m-3utr':data_dir + 'models/ntrans-v2-250m-3utr/checkpoints/epoch_23/'} 

# In[3]:

import argparse

parser = argparse.ArgumentParser("main.py")

parser.add_argument("--fasta", help = "fasta file name", type = str, required = True)

parser.add_argument("--model", help = "model name: NT-MS-v2-500M, DNABERT, DNABERT-2", type = str, required = True)

parser.add_argument("--output_dir", help = "output dir name", type = str, required = True)

parser.add_argument("--batch_size", help = "batch size", type = int, default = 10, required = False)

parser.add_argument("--N_folds", help = "number of folds to split sequences", type = int, default = None, required = False)

parser.add_argument("--max_tokenized_length", help = "force max tokenized length", type = int, default = None, required = False)

parser.add_argument("--fold", help = "current fold", type = int, default = None, required = False)

parser.add_argument("--include_txt", help = "list of sequences to include", type = str, default = None, required = False)

input_params = vars(parser.parse_args())

input_params = misc.dotdict(input_params)

for key,value in input_params.items():
    print(f'{key}: {value}')
    
if torch.cuda.is_available():
    device = torch.device('cuda')
    cuda_device_name = torch.cuda.get_device_name(0)
    print(f'\nCUDA device: {cuda_device_name}\n')
else:
    device = torch.device('cpu')
    print('\nCUDA device: CPU\n')
    #raise Exception('CUDA is not found')

max_length = {'dnabert':510,'dnabert2':512,'ntrans-v2-250m':1024,'ntrans-v2-500m':1024,
              'dnabert-3utr':510,'dnabert2-3utr':1024,'ntrans-v2-250m-3utr':1024,}


def load_model(model_name):

    print(f'Loading model {model_name} from {model_dirs[model_name]}')

    if  'dnabert' in model_name and not 'dnabert2' in model_name:
        
        tokenizer = DNATokenizer(vocab_file='./DNABERT/src/transformers/dnabert-config/bert-config-6/vocab.txt',max_len=510)
        model = BertForMaskedLM.from_pretrained(model_dirs[model_name]).to(device);

    elif 'dnabert2' in model_name:

        tokenizer = AutoTokenizer.from_pretrained(model_dirs[model_name],trust_remote_code=True)
        embeddings_model = AutoModel.from_pretrained(model_dirs[model_name],trust_remote_code=True).to(device);
        prediction_model = BertForMaskedLM.from_pretrained(model_dirs[model_name]).to(device);
        model = (embeddings_model, prediction_model)
    
    elif 'ntrans' in model_name:

        # Import the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(model_dirs[model_name],trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_dirs[model_name],trust_remote_code=True).to(device);

    return tokenizer, model

class SeqDataset(Dataset):
    
    def __init__(self, fasta_file, fold=None, N_folds=None, include_txt=None):
        
        seqs = defaultdict(str)
            
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    transcript_id = line[1:].rstrip()
                else:
                    seqs[transcript_id] += line.rstrip().upper()
                    
        #seqs = {k:v[:MAX_SEQ_LENGTH] for k,v in seqs.items()}
        #seqs = {k:''.join(np.random.choice(list('ACGT'),size=MAX_LENGTH)) for k,v in seqs.items()}
        seqs = list(seqs.items())

        if include_txt!=None:
            print(f'Including sequences from {include_txt}')
            processed_seqs = pd.read_csv(include_txt,names=['seq_name']).seq_name.values
            seqs = [(seq_name,seq) for seq_name,seq in seqs if seq_name in processed_seqs]
        if N_folds!=None:
            print(f'Fold {fold}')
            folds = np.tile(np.arange(N_folds),len(seqs)//N_folds+1)[:len(seqs)]
            seqs = [x for idx,x in enumerate(seqs) if folds[idx]==fold]
            
        self.seqs = seqs
        self.max_length = max([len(seq[1]) for seq in self.seqs])
        
    def __len__(self):
        
        return len(self.seqs)
    
    def __getitem__(self, idx):
        
        return self.seqs[idx]

def kmers_stride1(seq, k=6):
    # splits a sequence into overlapping k-mers
    return [seq[i:i + k] for i in range(0, len(seq)-k+1)] 

def get_batch_embeddings(model_name, sequences):

    if input_params.max_tokenized_length is None:
        max_tokenized_length = max_length[model_name]
    else:
        max_tokenized_length = input_params.max_tokenized_length

    if 'dnabert' in model_name and not 'dnabert2' in model_name:

        mean_sequence_embeddings = []
        losses = []

        #special_token_ids = [tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.unk_token_id]
        
        for seq in sequences:

            if len(seq)<6:
                emb_seq = np.zeros((1,768))
                emb_seq[:] = np.nan
                losses.append(np.nan)
                mean_sequence_embeddings.append(emb_seq)
                continue

            if len(seq)>max_tokenized_length:
                warnings.warn('Cutting out the central part of the sequence to center around DNABERT FOV')
                seq = misc.center_seq(seq,max_tokenized_length)

            seq_kmer = kmers_stride1(seq)
    
            tokenized_seq = tokenizer.encode_plus(seq_kmer, 
                                                  truncation = True, 
                                                  return_tensors = 'pt',  
                                                  add_special_tokens=True, 
                                                  padding='max_length', 
                                                  max_length=312).to(device)
                        
            torch_outs = model(tokenized_seq['input_ids'], 
                                labels = tokenized_seq['input_ids'],
                                attention_mask = tokenized_seq['attention_mask'], 
                                output_hidden_states=True)
            
            attention_mask = torch.unsqueeze(tokenized_seq['attention_mask'], dim=-1)
                    
            emb_seq = torch.sum(attention_mask*torch_outs.hidden_states[-1], axis=-2)/torch.sum(attention_mask, axis=1)
            emb_seq = emb_seq.detach().cpu().numpy()
      
            mean_sequence_embeddings.append(emb_seq)

            losses.append(torch_outs.loss.item())
            
            #probas = F.softmax(torch_outs['logits'],dim=2).detach().cpu().numpy()
            #token_ids = model_input.cpu().numpy()
            #gt_probas = np.take_along_axis(probas, token_ids[...,None], axis=2)
            #gt_probas = gt_probas[~np.isin(token_ids,special_token_ids)]
            #log_probas_seq = np.log(gt_probas).squeeze()
            #log_probas.append(log_probas_seq)

        return (np.vstack(mean_sequence_embeddings),np.array(losses))

    elif 'dnabert2' in model_name:
            
        inputs = tokenizer(sequences, 
                           truncation=True, 
                           return_tensors = 'pt', 
                           padding="max_length", 
                           max_length = max_tokenized_length).to(device)
        
        assert 'A100' in cuda_device_name, 'A100 GPU is required to generate embeddings for the DNABERT2 model'
            
        hidden_states = model[0](inputs["input_ids"],
                                attention_mask=inputs["attention_mask"])[0] # [1, sequence_length, 768]
        attention_mask = torch.unsqueeze(inputs['attention_mask'], dim=-1)
        
        # Compute mean embeddings per sequence
        mean_sequence_embeddings = torch.sum(attention_mask*hidden_states, axis=-2)/torch.sum(attention_mask, axis=1)
        
        #mean_sequence_embeddings = torch.mean(hidden_states, dim=1)# embedding with mean pooling
        
        torch_outs = model[1](inputs["input_ids"],
                            attention_mask=inputs["attention_mask"])

        losses = F.cross_entropy(torch_outs.logits.reshape((-1,torch_outs.logits.shape[-1])), 
                      inputs["input_ids"].reshape(-1), reduction='none').reshape((len(sequences),-1)).sum(1)
        
        losses = losses.detach().cpu().numpy()
        mean_sequence_embeddings = mean_sequence_embeddings.detach().cpu().numpy()

        return (mean_sequence_embeddings, losses)

    elif 'ntrans' in model_name:

        inputs = tokenizer.batch_encode_plus(sequences, 
                                                      truncation = True,
                                                      return_tensors="pt", 
                                                      padding="max_length", 
                                                      max_length = max_tokenized_length).to(device)
                    
        torch_outs = model(
            inputs["input_ids"],
            attention_mask=inputs['attention_mask'],
            encoder_attention_mask=inputs['attention_mask'],
            output_hidden_states=True)
        
        # Compute sequences embeddings
        embeddings = torch_outs['hidden_states'][-1]
        #print(f"Embeddings shape: {embeddings.shape}")
        #print(f"Embeddings per token: {embeddings}")
        
        # Add embed dimension axis
        attention_mask = torch.unsqueeze(inputs['attention_mask'], dim=-1)
        
        # Compute mean embeddings per sequence
        mean_sequence_embeddings = torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)
        #print(f"Mean sequence embeddings: {mean_sequence_embeddings}")

        #probas = F.softmax(torch_outs['logits'],dim=2).cpu().numpy()
        #inputs = inputs.cpu().numpy()
        #gt_probas = np.take_along_axis(probas, inputs[...,None], axis=2).squeeze()
        #log_probas = np.log(gt_probas)
        
        losses = F.cross_entropy(torch_outs.logits.reshape((-1,torch_outs.logits.shape[-1])), 
                      inputs["input_ids"].reshape(-1), reduction='none').reshape((len(sequences),-1)).sum(1)
        
        mean_sequence_embeddings = mean_sequence_embeddings.detach().cpu().numpy()
        losses = losses.detach().cpu().numpy()

    return (mean_sequence_embeddings, losses)

tokenizer, model = load_model(input_params.model)

dataset = SeqDataset(input_params.fasta, fold=input_params.fold, N_folds=input_params.N_folds, include_txt=input_params.include_txt)

dataloader = DataLoader(dataset = dataset, 
                        batch_size = input_params.batch_size, 
                        num_workers = 2, collate_fn = None, shuffle = False)


print(f'Total sequences: {len(dataset)}')

all_emb, all_losses = [], []

from helpers.misc import print

for seq_idx, (seq_names,sequences) in enumerate(dataloader):

    #if os.path.isfile(input_params.output_dir + f'/{seq_names[0]}.pickle'):
    #    continue
        
    print(f'generating embeddings for batch {seq_idx}/{len(dataloader)}')
    
    with torch.no_grad():
        embeddings, losses = get_batch_embeddings(input_params.model,sequences)

    all_emb.append(embeddings)
    all_losses.append(losses)

    #with open(input_params.output_dir + f'/{seq_names[0]}.pickle', 'wb') as f:
    #    pickle.dump((seq_names,emb,logprobs),f)

if input_params.fold!=None:
    output_name = input_params.output_dir + f'/predictions_{input_params.fold}.pickle'
else:
    output_name = input_params.output_dir + '/predictions.pickle'

os.makedirs(input_params.output_dir, exist_ok=True)

seq_names, seqs = zip(*dataset.seqs)

with open(output_name, 'wb') as f:
    pickle.dump({'seq_names':seq_names, 'seqs':seqs, 'embeddings':np.vstack(all_emb), 
                 'losses':np.hstack(all_losses), 'fasta':input_params.fasta},f)
