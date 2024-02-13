#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import argparse
import json
import pickle
import torch.distributed as dist
from builtins import print

import os
import gc

import pysam

#
#install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 torchtriton -c pytorch -c nvidia
#pip install triton==2.0.0.dev20221202 --force --no-dependencies
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from DNABERT2.configuration_bert import BertConfig
from DNABERT2.bert_layers import BertForMaskedLM
import os

from tqdm import tqdm

import helpers.misc as misc                #miscellaneous functions
import helpers.train_eval as train_eval    #train and evaluation

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

parser = argparse.ArgumentParser("main.py")

parser.add_argument("--fasta", help = "FASTA file", type = str, required = True)
parser.add_argument("--output_dir", help = "dir to save predictions and model/optimizer weights", type = str, required = True)
parser.add_argument("--checkpoint_dir", help = "initialization weight of the model", type = str, default = None, required = False)
parser.add_argument("--val_fraction", help = "fraction of validation dataset to use", type = float, default = 0.1, required = False)
parser.add_argument("--validate_every", help = "validate every N epochs", type = int,  default = 1, required = False)
parser.add_argument("--species_agnostic", help = "use a pecies agnostic version", action='store_true', default = False, required = False)
parser.add_argument("--train_chunks", help = "split each epoch into N epochs", type = int, default = 8, required = False)
parser.add_argument("--tot_epochs", help = "total number of training epochs, (after splitting)", type = int, default = 20, required = False)
parser.add_argument("--batch_size", help = "batch size", type = int, default = 1024, required = False)
parser.add_argument("--max_lr", help = "learning rate", type = float, default = 5e-4, required = False)
parser.add_argument("--weight_decay", help = "Adam weight decay", type = float, default = 1e-5, required = False)
parser.add_argument("--save_at", help = "epochs to save model/optimizer weights, 1-based", nargs='+', type = str, default = ['1:30'], required = False)

input_params = vars(parser.parse_args())

#def main(input_params):

input_params = misc.dotdict(input_params)

input_params.save_at = misc.list2range(input_params.save_at)

# In[4]:
assert "WORLD_SIZE" in os.environ
assert 'SLURM_PROCID' in os.environ # for slurm scheduler

input_params.world_size = int(os.environ["WORLD_SIZE"])
ngpus_per_node = torch.cuda.device_count()
input_params.rank = int(os.environ['SLURM_PROCID'])
input_params.gpu = input_params.rank % torch.cuda.device_count()

print(f'tot gpus:{ngpus_per_node} world size:{input_params.world_size}, rank:{input_params.rank}, gpu:{input_params.gpu}')

dist.init_process_group(backend='nccl', world_size=input_params.world_size, rank=input_params.rank,
                       init_method='file://'+input_params.output_dir+'/sharedfile')

if input_params.rank==0:
    for param_name in ['output_dir', '\\',
    'fasta', 'species_agnostic', '\\',
    'tot_epochs', 'train_chunks', 'batch_size', 'save_at', '\\',
    'val_fraction', 'validate_every', '\\',
    'checkpoint_dir', '\\',
    'max_lr', 'weight_decay', '\\',
    ]:

        if param_name == '\\':
            print()
        else:
            print(f'{param_name.upper()}: {input_params[param_name]}')

# In[5]:

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

            tokenized_seq = tokenizer(seq,
                                    add_special_tokens=True,
                                    truncation=True,
                                    return_special_tokens_mask=True,
                                    padding='max_length',
                                    max_length=input_params.max_tok_len,)

            tokenized_seq['targets'] = tokenized_seq['input_ids'].copy()
            yield tokenized_seq

    def close(self):
        self.fasta.close()

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

#input_params.world_size = 8#dist.get_world_size()
#input_params.rank = 2#dist.get_rank()

assert input_params.batch_size % input_params.world_size == 0, 'batch size should be divisible by world size'

seq_df = pd.read_csv(input_params.fasta + '.fai', header=None, sep='\t', usecols=[0], names=['seq_name'])

#seq_df = seq_df.iloc[:3000]

seq_df['species_name'] = seq_df.seq_name.apply(lambda x:x.split(':')).apply(lambda x:x[-1] if len(x)==2 else x[1])

all_species = sorted(seq_df.species_name.unique())

if not input_params.species_agnostic:
    species_encoding = {species:idx for idx,species in enumerate(all_species)}
else:
    species_encoding = {species:0 for species in all_species}

seq_df['species_label'] = seq_df.species_name.map(species_encoding)

#seq_df = seq_df.sample(frac = 1., random_state = 1) #DO NOT SHUFFLE, otherwise too slow

tokenizer = PreTrainedTokenizerFast(tokenizer_file="../DNABERT2/tokenizer.json",
mask_token = '[MASK]', pad_token = '[PAD]', sep_token = '[SEP]', cls_token = '[CLS]', unk_token = '[UNK]',)
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

N_train = int(len(seq_df)*(1-input_params.val_fraction))

num_samples_per_rank = N_train // input_params.world_size

train_idx = range(input_params.rank*num_samples_per_rank,(input_params.rank+1)*num_samples_per_rank)

print(f'Rank:{input_params.rank} train index:{train_idx[0]}-{train_idx[-1]}')

train_df = seq_df.iloc[train_idx].reset_index(drop=True)

train_chunk = np.repeat(list(range(input_params.train_chunks)),repeats = len(train_df) // input_params.train_chunks + 1 )
train_df['train_chunk'] = train_chunk[:len(train_df)]

test_df = seq_df.iloc[num_samples_per_rank*input_params.world_size:]

train_dataset = SeqDataset(input_params.fasta, train_df)
train_dataloader = DataLoader(dataset = train_dataset, batch_size = input_params.batch_size // input_params.world_size,
                              num_workers = 1, worker_init_fn=worker_init_fn, collate_fn = collate_fn, shuffle = False)

test_dataset = SeqDataset(input_params.fasta, test_df)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = input_params.batch_size // input_params.world_size,
                             num_workers = 1,  worker_init_fn=worker_init_fn, collate_fn = collate_fn, shuffle = False)

torch.backends.cudnn.benchmark = True

gc.collect()
torch.cuda.empty_cache()


config = BertConfig.from_pretrained('../DNABERT2/config.json')

model = BertForMaskedLM(config)

torch.cuda.set_device(input_params.gpu)

model.cuda(input_params.gpu)

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[input_params.gpu])

model_params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(model_params, betas=(0.9,0.98), eps=1e-6,
                             lr = input_params.max_lr,
                             weight_decay = input_params.weight_decay)

last_epoch, last_iteration = 0, -1

if input_params.checkpoint_dir:

    model = BertForMaskedLM.from_pretrained(input_params.checkpoint_dir).to(device)

    if os.path.isfile(input_params.checkpoint_dir + '/opimizer.pt'):
            optimizer.load_state_dict(torch.load(input_params.checkpoint_dir + '/opimizer.pt'))
            scheduler.load_state_dict(torch.load(input_params.checkpoint_dir + '/scheduler.pt'))

weights_dir = os.path.join(input_params.output_dir, 'checkpoints') #dir to save model weights at save_at epochs

if input_params.save_at:
    os.makedirs(weights_dir, exist_ok = True)


scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 0, max_lr = input_params.max_lr,
                                             step_size_up = 20000, step_size_down = 180000, cycle_momentum=False,
                                             last_epoch = last_iteration)

def metrics_to_str(metrics):
    loss, total_acc, masked_acc = metrics
    return f'loss: {loss:.4}, total acc: {total_acc:.3f}, masked acc: {masked_acc:.3f}'


# if input_params.rank == 0:
#    from utils.misc import print_log as print    #print function that displays time
# else:
#    from utils.misc import print_pass as print

from utils.misc import print_log as print

if not input_params.test:

    for epoch in range(last_epoch+1, input_params.tot_epochs+1):

        train_dataset.seq_df = train_df[train_df.train_chunk == (epoch-1) % input_params.train_chunks]
        train_dataset.end = len(train_dataset.seq_df)

        print(f'EPOCH {epoch}: Training; rank {input_params.rank}, using train samples: {list(train_dataset.seq_df.index[[0,-1]])}')

        train_metrics = train_eval.model_train(model, optimizer, train_dataloader, input_params.gpu, scheduler=scheduler,
                            silent = True)

        last_iteration+=int(np.ceil(len(train_dataset)/train_dataloader.batch_size))

        if input_params.rank == 0:
            print(f'epoch {epoch} - train ({last_iteration+1} iterations), {metrics_to_str(train_metrics)}')

        if epoch in input_params.save_at and input_params.rank == 0: #save model weights

            misc.save_model_weights(model, optimizer, scheduler, weights_dir, epoch)

        if input_params.val_fraction>0 and input_params.rank == 0 and ( epoch==input_params.tot_epochs or
                            (input_params.validate_every and epoch%input_params.validate_every==0)):

            print(f'EPOCH {epoch}: Validating...')

            val_metrics =  train_eval.model_eval(model, optimizer, test_dataloader, input_params.gpu,
                    silent = True)

            print(f'epoch {epoch} - validation, {metrics_to_str(val_metrics)}')


# if __name__ == '__main__':
#     input_params = vars(parser.parse_args())
#     main(input_params)

# In[ ]:
