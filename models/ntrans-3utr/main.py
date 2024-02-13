#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import pickle
import argparse
import json

import numpy as np
import pandas as pd

from tqdm import tqdm

import pysam

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_inverse_sqrt_schedule

from ntrans.esm_config import EsmConfig
from ntrans.modeling_esm import EsmForMaskedLM

import helpers.misc as misc                #miscellaneous functions
import helpers.train_eval as train_eval    #train and evaluation


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


# In[3]:


if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'\nCUDA device: {torch.cuda.get_device_name(0)}\n')
else:
    device = torch.device('cpu')
    print('\nCUDA device: CPU\n')
    #raise Exception('CUDA is not found')


class SeqDataset(IterableDataset):
    
    def __init__(self, fasta_fa, seq_df):
        
        if fasta_fa:
            self.fasta = pysam.FastaFile(fasta_fa)
        else:
             self.fasta = None

        self.seq_df = seq_df
        self.start = 0
        self.end = len(self.seq_df)
        
    def __len__(self):
        return len(self.seq_df)
                
    def __iter__(self):
        
        #worker_total_num = torch.utils.data.get_worker_info().num_workers
        #worker_id = torch.utils.data.get_worker_info().id
        
        for seq_idx in range(self.start,self.end):
            
            if self.fasta:
                seq = self.fasta.fetch(self.seq_df.iloc[seq_idx].seq_name).upper()
            else:
                seq = self.seq_df.iloc[seq_idx].seq.upper()
    
            #species_label = self.seq_df.iloc[idx].species_label
            
            seq = seq.replace('-','')

            tokenized_seq = tokenizer(seq, add_special_tokens=False)['input_ids']

            #N_tokens_overlap=np.random.randint(low=0,high=input_params.max_overlap_tokens),

            tokenized_chunks, _ = misc.get_chunks(tokenized_seq, 
                                                   N_tokens_chunk=input_params.max_tokens, 
                                                   N_tokens_overlap=input_params.max_overlap_tokens,
                                                   tokenizer_cls_token_id=tokenizer.cls_token_id,
                                                   tokenizer_eos_token_id=None,
                                                   tokenizer_pad_token_id=None,
                                                   padding=False)

            for tokenized_chunk in tokenized_chunks:

                attention_mask = [1 if token_id!=tokenizer.pad_token_id else 0 for token_id in tokenized_chunk]
                
                tokenized_chunk = {'input_ids':tokenized_chunk,
                                   'seq_idx':seq_idx,
                                   'token_type_ids':[0]*len(tokenized_chunk), 
                                   'attention_mask':attention_mask}
                
                yield tokenized_chunk
                        
    def close(self):
        self.fasta.close()

# In[6]:

parser = argparse.ArgumentParser("main.py")

parser.add_argument("--fasta", help = "FASTA file", type = str, required = True)
parser.add_argument("--output_dir", help = "dir to save predictions and model/optimizer weights", type = str, required = True)
parser.add_argument("--checkpoint_dir", help = "initialization weight of the model", type = str, default = None, required = False)
parser.add_argument("--val_fraction", help = "fraction of validation dataset to use", type = float, default = 0.1, required = False)
parser.add_argument("--validate_every", help = "validate every N epochs", type = int,  default = 1, required = False)
parser.add_argument("--model_name", help = "model name, e.g. v2-500M", type=str, required = True)
parser.add_argument("--species_agnostic", help = "use a species agnostic version", action='store_true', default = False, required = False)
parser.add_argument("--train_chunks", help = "split each epoch into N epochs", type = int, default = 8, required = False)
parser.add_argument("--max_overlap_tokens", help = "maximal tokenized chunk overlap", type = int, default = 50, required = False)
parser.add_argument("--max_tokens", help = "maximal chunk length in tokens", type = int, default = 1024, required = False)
parser.add_argument("--mlm_probability", help = "masking probability", type = float, default = 0.15, required = False)
parser.add_argument("--tot_epochs", help = "total number of training epochs, (after splitting)", type = int, default = 20, required = False)
parser.add_argument("--batch_size", help = "batch size", type = int, default = 16, required = False)
parser.add_argument("--max_lr", help = "learning rate", type = float, default = 1e-4, required = False)
parser.add_argument("--weight_decay", help = "Adam weight decay", type = float, default = 0, required = False)
parser.add_argument("--save_at", help = "epochs to save model/optimizer weights, 1-based", nargs='+', type = str, default = [], required = False)
parser.add_argument("--step_size_up", help = "number of iterations for lr ascending", type = int, default = 16000, required = False)
parser.add_argument("--step_size_down", help = "number of iterations for lr descending", type = int, default = None, required = False)

input_params = vars(parser.parse_args())

input_params = misc.dotdict(input_params)

input_params.save_at = misc.list2range(input_params.save_at)

for param_name in ['output_dir', '\\',
        'fasta', 'checkpoint_dir', '\\',
        'model_name','species_agnostic', '\\',
        'tot_epochs', 'save_at', '\\',
        'train_chunks','val_fraction', 'validate_every', '\\',
        'max_overlap_tokens','max_tokens', '\\',
        'mlm_probability', '\\',
        'batch_size', 'max_lr', 'weight_decay', 'step_size_up','step_size_down', '\\',
        ]:

            if param_name == '\\':
                print()
            else:
                print(f'{param_name.upper()}: {input_params[param_name]}')

seq_df = pd.read_csv(input_params.fasta + '.fai', header=None, sep='\t', usecols=[0,1], names=['seq_name','seq_len'])

seq_df['species_name'] = seq_df.seq_name.apply(lambda x:x.split(':')).apply(lambda x:x[1] if len(x)==2 else 'Homo_sapiens')

all_species = sorted(seq_df.species_name.unique())

if not input_params.species_agnostic:
    species_encoding = {species:idx for idx,species in enumerate(all_species)}
else:
    species_encoding = {species:0 for species in all_species}

seq_df['species_label'] = seq_df.species_name.map(species_encoding)

#seq_df = seq_df.sample(frac = 1., random_state = 1) #DO NOT SHUFFLE, otherwise too slow


# In[8]:


tokenizer = AutoTokenizer.from_pretrained('3utr_tokenizer',model_max_length=input_params.max_tok_len)

tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=input_params.mlm_probability)


# In[9]:


N_train = int(len(seq_df)*(1-input_params.val_fraction))
train_df, test_df = seq_df.iloc[:N_train], seq_df.iloc[N_train:]

train_chunk = np.repeat(list(range(input_params.train_chunks)),repeats = N_train // input_params.train_chunks + 1 )
train_df['train_chunk'] = train_chunk[:N_train]

train_dataset = SeqDataset(input_params.fasta, train_df)
train_dataloader = DataLoader(dataset = train_dataset, batch_size = input_params.batch_size,
                              num_workers = 1, worker_init_fn=misc.worker_init_fn, collate_fn = collate_fn, shuffle = False)

test_dataset = SeqDataset(input_params.fasta, test_df)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = input_params.batch_size,
                             num_workers = 1,  worker_init_fn=misc.worker_init_fn, collate_fn = collate_fn, shuffle = False)


# In[10]:


gc.collect()
torch.cuda.empty_cache()


# In[11]:



config = EsmConfig.from_pretrained(f'ntrans/config_{input_params.model_name}.json')

config.vocab_size = len(tokenizer)

model = EsmForMaskedLM(config).to(device)

model.resize_token_embeddings(len(tokenizer))


# In[12]:


model_params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(model_params, betas=(0.9,0.999), eps=1e-8,
                             lr = input_params.max_lr,
                             weight_decay = input_params.weight_decay)


# In[13]:

if input_params.step_size_down is not None:
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 0, max_lr = input_params.max_lr,
        step_size_up = input_params.step_size_up, step_size_down = input_params.step_size_down, cycle_momentum=False)
else:
    scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=input_params.step_size_up,)

last_epoch = 0

if input_params.checkpoint_dir:

    model = BertForMaskedLM.from_pretrained(input_params.checkpoint_dir).to(device)

    tokenizer = EsmTokenizer.from_pretrained(input_params.checkpoint_dir)

    if os.path.isfile(input_params.checkpoint_dir + '/optimizer.pt'):
            optimizer.load_state_dict(torch.load(input_params.checkpoint_dir + '/optimizer.pt'))
            scheduler.load_state_dict(torch.load(input_params.checkpoint_dir + '/scheduler.pt'))

    last_epoch = int(input_params.checkpoint_dir.rstrip('/').split('_')[-1]) #infer previous epoch from input_params.checkpoint_dir


weights_dir = os.path.join(input_params.output_dir, 'checkpoints') #dir to save model weights at save_at epochs

if input_params.save_at:
    os.makedirs(weights_dir, exist_ok = True)


def metrics_to_str(metrics):
    loss, total_acc, masked_acc = metrics
    return f'loss: {loss:.4}, total acc: {total_acc:.3f}, masked acc: {masked_acc:.3f}'

from helpers.misc import print    #print function that displays time

if not input_params.test:

    for epoch in range(last_epoch+1, input_params.tot_epochs+1):

        print(f'EPOCH {epoch}: Training...')

        train_dataset.seq_df = train_df[train_df.train_chunk == (epoch-1) % input_params.train_chunks]
        train_dataset.end = len(train_dataset.seq_df)

        print(f'using train samples: {list(train_dataset.seq_df.index[[0,-1]])}')

        train_metrics = train_eval.model_train(model, optimizer, train_dataloader, device, scheduler=scheduler,
                            silent = True)

        print(f'epoch {epoch} - train ({scheduler.last_epoch+1} iterations), {metrics_to_str(train_metrics)}')

        if epoch in input_params.save_at or -1 in input_params.save_at: #save model weights

            checkpoint_dir = misc.save_model_weights(model, None, optimizer, scheduler, weights_dir, epoch, input_params.save_at)
            _ = os.system('cp ./ntrans/*.py ' + checkpoint_dir)
            _ = os.system('cp ./3utr_tokenizer/* ' + checkpoint_dir) 
            
        if input_params.val_fraction>0 and ( epoch==input_params.tot_epochs or
                            (input_params.validate_every and epoch%input_params.validate_every==0)):

            print(f'EPOCH {epoch}: Validating...')

            val_metrics =  train_eval.model_eval(model, optimizer, test_dataloader, device,
                    silent = True)

            print(f'epoch {epoch} - validation, {metrics_to_str(val_metrics)}')



# In[ ]:
